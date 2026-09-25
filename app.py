import os
import io
import base64
import logging
import time as _time
from functools import wraps
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, render_template,
    redirect, flash, url_for, send_file, session
)
from werkzeug.utils import secure_filename
from sqlalchemy import func
import pandas as pd
from io import BytesIO
from xhtml2pdf import pisa

from database import db
from face_service import (
    serialize_encoding, deserialize_encoding, match_faces,
    EncodingCache, downscale_factor
)
from demo import init_demo, ensure_demo_data
from models import (
    Employee, Attendance, Leave, SystemSettings,
    User, FaceEncoding, RecognitionLog
)

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # reject uploads/frames > 8 MB
log = logging.getLogger('attendanceai')

# Database config — SQLite by default, override with DATABASE_URL in production
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Match confidence: how close a live face must be to a stored one to count as a match
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
# Check-ins after this local time are marked 'Late' (HH:MM, override with LATE_CUTOFF env var)
LATE_CUTOFF = datetime.strptime(os.environ.get('LATE_CUTOFF', '09:15'), '%H:%M').time()
DEFAULT_TIMEZONE = 'Asia/Kolkata'

if app.secret_key == 'dev-key-change-in-production' or ADMIN_PASSWORD == 'admin123':
    log.warning('Running with default SECRET_KEY / ADMIN_PASSWORD - set them in the environment before deploying.')

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


# ---------------------- Auth Helpers ----------------------

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def ensure_admin_user():
    if not User.query.filter_by(username=ADMIN_USERNAME).first():
        admin = User(username=ADMIN_USERNAME, role='admin')
        admin.set_password(ADMIN_PASSWORD)
        db.session.add(admin)
        db.session.commit()


def ensure_settings_row():
    if not SystemSettings.query.first():
        db.session.add(SystemSettings(
            company_name='My Company',
            timezone='Asia/Kolkata',
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
        ))
        db.session.commit()


# ---------------------- Time Helpers ----------------------
# Servers (e.g. Render) run in UTC. Attendance must be recorded in the company's
# local time, otherwise an 09:05 IST check-in is stored as 03:35 and "today"
# rolls over at 05:30 IST.

def company_tz():
    row = SystemSettings.query.first()
    name = (row.timezone if row and row.timezone else DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_TIMEZONE)


def now_local():
    """Current wall-clock time in the company timezone (naive, for storage)."""
    return datetime.now(company_tz()).replace(tzinfo=None)


def today_local():
    return now_local().date()


def day_bounds(day):
    """[start, end) datetimes for a calendar day - portable across SQLite/Postgres."""
    start = datetime.combine(day, time.min)
    return start, start + timedelta(days=1)


def status_for_checkin(check_in_dt):
    return 'Late' if check_in_dt.time() > LATE_CUTOFF else 'Present'


# ---------------------- Face Recognition Helpers ----------------------

encoding_cache = EncodingCache()


def compute_face_encoding(image_path):
    """Return the 128-d encoding of the largest face in an image, or None."""
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    if not locations:
        return None
    # If the photo has several people in it, enrol the biggest (closest) face.
    largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
    encodings = face_recognition.face_encodings(image, [largest])
    return encodings[0] if encodings else None


def _load_enrolled():
    ids, names, vectors = [], [], []
    for row in FaceEncoding.query.all():
        try:
            vec = deserialize_encoding(row.encoding)
        except Exception:
            continue
        if vec is not None and row.employee is not None:
            ids.append(row.employee.id)
            names.append(row.employee.name)
            vectors.append(vec)
    return ids, names, vectors


def save_face_encoding(employee, encoding):
    blob = serialize_encoding(encoding)
    if employee.face_encoding:
        employee.face_encoding.encoding = blob
    else:
        db.session.add(FaceEncoding(employee_id=employee.id, encoding=blob))
    db.session.commit()
    encoding_cache.invalidate()


def decode_base64_image(data_url):
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img_bytes = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return image


def detect_and_encode(image):
    """
    Detect + encode every face in a PIL image.
    Frames wider than 640px are downscaled first (detection cost grows with
    pixel count); boxes are scaled back to the original frame for the UI overlay.
    Returns (encodings, boxes) where boxes are {top,right,bottom,left} in original px.
    """
    scale = downscale_factor(image.width)
    small = image if scale == 1.0 else image.resize(
        (int(image.width * scale), int(image.height * scale)))
    frame = np.array(small)
    locations = face_recognition.face_locations(frame)
    if not locations:
        return [], []
    encodings = face_recognition.face_encodings(frame, locations)
    boxes = [{'top': int(t / scale), 'right': int(r / scale),
              'bottom': int(b / scale), 'left': int(l / scale)}
             for (t, r, b, l) in locations]
    return encodings, boxes


def mark_attendance_checkin(employee_id, source='face_recognition'):
    """Create today's check-in (local time) if one doesn't exist yet."""
    now = now_local()
    start, end = day_bounds(now.date())
    existing = Attendance.query.filter(
        Attendance.employee_id == employee_id,
        Attendance.check_in >= start, Attendance.check_in < end
    ).first()
    if existing:
        return False
    db.session.add(Attendance(
        employee_id=employee_id,
        check_in=now,
        status=status_for_checkin(now),
        source=source
    ))
    db.session.commit()
    return True


# ---------------------- Routes ----------------------

@app.route('/')
def index():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.')
        return render_template('login.html')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    total_employees = Employee.query.count()
    total_leaves = Leave.query.filter_by(status='Approved').count()
    total_records = Attendance.query.count()
    # Late arrivals still attended - count them as present for the rate
    present_records = Attendance.query.filter(Attendance.status.in_(['Present', 'Late'])).count()
    attendance_rate = round((present_records / total_records) * 100, 2) if total_records else 0
    late_arrivals = Attendance.query.filter(Attendance.status == 'Late').count()

    recent_attendance = Attendance.query.order_by(Attendance.check_in.desc()).limit(5).all()
    recent_leaves = Leave.query.order_by(Leave.from_date.desc()).limit(5).all()
    recent_recognitions = RecognitionLog.query.order_by(RecognitionLog.timestamp.desc()).limit(5).all()

    return render_template(
        'dashboard.html',
        total_employees=total_employees,
        total_leaves=total_leaves,
        attendance_rate=attendance_rate,
        late_arrivals=late_arrivals,
        recent_attendance=recent_attendance,
        recent_leaves=recent_leaves,
        recent_recognitions=recent_recognitions
    )


# ------------------ Employee Management ------------------

@app.route('/employees', methods=['GET', 'POST'])
@login_required
def manage_employees():
    if request.method == 'POST':
        data = request.json
        new_employee = Employee(
            employee_id=data['employee_id'],
            name=data['name'],
            email=data['email'],
            phone=data['phone'],
            department=data['department'],
            role=data['role'],
            date_of_joining=data['date_of_joining']
        )
        db.session.add(new_employee)
        db.session.commit()
        return jsonify({'message': 'Employee added successfully!'}), 201

    employees = Employee.query.all()
    return render_template('employees.html', employees=employees)


@app.route('/employees/add', methods=['GET', 'POST'])
@login_required
def add_employee():
    if request.method == 'POST':
        date_of_joining = datetime.strptime(request.form['date_of_joining'], '%Y-%m-%d').date()

        if Employee.query.filter_by(employee_id=request.form['employee_id']).first():
            flash(f"Employee ID '{request.form['employee_id']}' is already in use — pick a different one.")
            return render_template('add_employee.html')

        new_employee = Employee(
            employee_id=request.form['employee_id'],
            name=request.form['name'],
            email=request.form['email'],
            phone=request.form['phone'],
            department=request.form['department'],
            role=request.form['role'],
            date_of_joining=date_of_joining
        )
        db.session.add(new_employee)
        db.session.commit()

        photo = request.files.get('photo')
        if photo and photo.filename:
            filename = secure_filename(f"{new_employee.employee_id}_{photo.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(filepath)
            new_employee.photo_path = os.path.join('uploads', filename)
            db.session.commit()

            if FACE_RECOGNITION_AVAILABLE:
                encoding = compute_face_encoding(filepath)
                if encoding is not None:
                    save_face_encoding(new_employee, encoding)
                    flash(f'{new_employee.name} added — face registered for recognition.')
                else:
                    flash(f'{new_employee.name} added, but no face was detected in that photo. Upload a clearer front-facing photo from the edit page to enable recognition.')
            else:
                flash(f'{new_employee.name} added. Face recognition libraries are not installed on this server.')
        else:
            flash(f'{new_employee.name} added without a photo — recognition is off for them until one is uploaded.')

        return redirect('/employees')
    return render_template('add_employee.html')


@app.route('/employees/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_employee(id):
    employee = db.get_or_404(Employee, id)
    if request.method == 'POST':
        clash = Employee.query.filter(Employee.employee_id == request.form['employee_id'],
                                      Employee.id != employee.id).first()
        if clash:
            flash(f"Employee ID '{request.form['employee_id']}' belongs to {clash.name} - pick a different one.")
            return render_template('edit_employee.html', employee=employee)
        employee.employee_id = request.form['employee_id']
        employee.name = request.form['name']
        employee.email = request.form['email']
        employee.phone = request.form['phone']
        employee.department = request.form['department']
        employee.role = request.form['role']
        employee.date_of_joining = datetime.strptime(request.form['date_of_joining'], '%Y-%m-%d').date()
        db.session.commit()

        photo = request.files.get('photo')
        if photo and photo.filename:
            filename = secure_filename(f"{employee.employee_id}_{photo.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(filepath)
            employee.photo_path = os.path.join('uploads', filename)
            db.session.commit()

            if FACE_RECOGNITION_AVAILABLE:
                encoding = compute_face_encoding(filepath)
                if encoding is not None:
                    save_face_encoding(employee, encoding)
                    flash('Photo updated and face re-registered.')
                else:
                    flash('Photo updated, but no face was detected in it.')

        return redirect(url_for('manage_employees'))
    return render_template('edit_employee.html', employee=employee)


@app.route('/employees/delete/<int:id>')
@login_required
def delete_employee(id):
    employee = db.get_or_404(Employee, id)
    db.session.delete(employee)
    db.session.commit()
    encoding_cache.invalidate()
    return redirect(url_for('manage_employees'))


# ------------------ Attendance ------------------

@app.route('/attendance')
@login_required
def attendance():
    employees = Employee.query.all()
    attendance_records = Attendance.query.order_by(Attendance.check_in.desc()).all()
    return render_template('attendance.html', employees=employees, attendance_records=attendance_records)


@app.route('/attendance/check_in/<int:emp_id>')
@login_required
def check_in(emp_id):
    if not mark_attendance_checkin(emp_id, source='manual'):
        flash('Already checked in today.')
    return redirect(url_for('attendance'))


@app.route('/attendance/check_out/<int:emp_id>')
@login_required
def check_out(emp_id):
    start, end = day_bounds(today_local())
    record = Attendance.query.filter(Attendance.employee_id == emp_id,
                                     Attendance.check_in >= start,
                                     Attendance.check_in < end).first()
    if record and not record.check_out:
        record.check_out = now_local()
        db.session.commit()
    return redirect(url_for('attendance'))


# ------------------ Face Recognition ------------------

@app.route('/face_recognition')
@login_required
def face_recognition_page():
    settings_row = SystemSettings.query.first()
    recognition_logs = RecognitionLog.query.order_by(RecognitionLog.timestamp.desc()).limit(20).all()
    registered_count = FaceEncoding.query.count()
    total_employees = Employee.query.count()
    return render_template(
        'face_recognition.html',
        settings=settings_row,
        recognition_logs=recognition_logs,
        registered_count=registered_count,
        total_employees=total_employees,
        library_available=FACE_RECOGNITION_AVAILABLE
    )


@app.route('/api/recognize-face', methods=['POST'])
@login_required
def recognize_face():
    """
    Recognise every face in a webcam frame.

    Normal session : matches against enrolled employees, logs the attempt and
                     checks matched employees in (once per day).
    Demo session   : dry run - nothing is written. The visitor's own face (if they
                     enrolled it via /api/demo/enroll) is matched too, so recruiters
                     can see recognition work on themselves.
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({'status': 'error', 'message': 'face_recognition library is not installed on this server'}), 503

    data = request.get_json(silent=True) or {}
    image_data = data.get('image')
    camera_name = str(data.get('camera', 'Default Camera'))[:100]
    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    settings_row = SystemSettings.query.first()
    threshold = settings_row.confidence_threshold if settings_row else DEFAULT_CONFIDENCE_THRESHOLD

    try:
        image = decode_base64_image(image_data)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

    t0 = _time.perf_counter()
    unknown_encodings, boxes = detect_and_encode(image)
    if not unknown_encodings:
        return jsonify({'status': 'no_face', 'message': 'No face detected in frame',
                        'faces': [], 'processing_ms': round((_time.perf_counter() - t0) * 1000)})

    ids, names, matrix = encoding_cache.get(_load_enrolled)
    is_demo = bool(session.get('demo'))
    if is_demo and session.get('demo_face'):
        ids = list(ids) + [None]
        names = list(names) + [session.get('demo_face_name', 'You')]
        matrix = np.vstack([matrix, np.asarray(session['demo_face'], dtype=np.float64)])

    matches = match_faces(unknown_encodings, matrix, threshold)
    processing_ms = round((_time.perf_counter() - t0) * 1000)

    faces = []
    for match, box in zip(matches, boxes):
        idx = match['index']
        faces.append({
            'box': box,
            'employee_id': ids[idx] if idx is not None else None,
            'name': names[idx] if idx is not None else 'Unknown',
            'confidence': round(match['confidence'] * 100, 1),
            'status': 'success' if idx is not None else 'unrecognized',
            'checked_in': False,
        })

    if not is_demo:
        for face in faces:
            db.session.add(RecognitionLog(
                employee_id=face['employee_id'],
                employee_name=face['name'],
                confidence=face['confidence'],
                camera_name=camera_name,
                status='Success' if face['employee_id'] else 'Unrecognized',
                timestamp=now_local()
            ))
        db.session.commit()
        for face in faces:
            if face['employee_id']:
                face['checked_in'] = mark_attendance_checkin(face['employee_id'])

    # Top-level fields describe the best face, for backward compatibility with v1 clients
    best = max(faces, key=lambda f: (f['status'] == 'success', f['confidence']))
    return jsonify({
        'status': best['status'],
        'name': best['name'],
        'confidence': best['confidence'],
        'checked_in': best['checked_in'],
        'faces': faces,
        'processing_ms': processing_ms,
        'demo': is_demo,
    })


@app.route('/api/demo/enroll', methods=['POST'])
@login_required
def demo_enroll():
    """Demo only: enrol the visitor's face in their session (never stored in the DB)."""
    if not session.get('demo'):
        return jsonify({'status': 'error', 'message': 'Only available in demo mode'}), 403
    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({'status': 'error', 'message': 'face_recognition library is not installed on this server'}), 503
    data = request.get_json(silent=True) or {}
    try:
        image = decode_base64_image(data.get('image') or '')
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400
    encodings, boxes = detect_and_encode(image)
    if len(encodings) != 1:
        msg = 'No face found - look at the camera.' if not encodings else 'Several faces found - only you in frame, please.'
        return jsonify({'status': 'error', 'message': msg}), 422
    name = (str(data.get('name') or 'You').strip() or 'You')[:40]
    session['demo_face'] = [round(float(v), 6) for v in encodings[0]]
    session['demo_face_name'] = name
    return jsonify({'status': 'enrolled', 'name': name, 'box': boxes[0]})


@app.route('/api/demo/reset', methods=['POST'])
@login_required
def demo_reset():
    session.pop('demo_face', None)
    session.pop('demo_face_name', None)
    return jsonify({'status': 'reset'})


@app.route('/healthz')
def healthz():
    """Uptime check for Render / monitoring."""
    return jsonify({'status': 'ok', 'face_recognition': FACE_RECOGNITION_AVAILABLE})


# ------------------ Reports ------------------

def resolve_report_day(raw):
    """?date=YYYY-MM-DD, else today; if today has no records yet (early morning,
    weekends, demo data), fall back to the most recent day that does."""
    if raw:
        try:
            return datetime.strptime(raw, '%Y-%m-%d').date()
        except ValueError:
            pass
    day = today_local()
    start, end = day_bounds(day)
    if Attendance.query.filter(Attendance.check_in >= start, Attendance.check_in < end).first():
        return day
    latest = Attendance.query.filter(Attendance.check_in.isnot(None)) \
        .order_by(Attendance.check_in.desc()).first()
    return latest.check_in.date() if latest else day


def build_department_report(day):
    """
    One source of truth for the Reports page, PDF and Excel exports
    (v1 computed these three differently, so the PDF always showed 0 late arrivals).
    Two queries total instead of one query per employee.
    """
    start, end = day_bounds(day)
    headcount = dict(db.session.query(Employee.department, func.count(Employee.id))
                     .group_by(Employee.department).all())
    rows = db.session.query(Employee.department, Attendance.status) \
        .join(Attendance, Attendance.employee_id == Employee.id) \
        .filter(Attendance.check_in >= start, Attendance.check_in < end).all()

    present, late = {}, {}
    for dept, status in rows:
        present[dept] = present.get(dept, 0) + 1
        if status == 'Late':
            late[dept] = late.get(dept, 0) + 1

    report = []
    for dept in sorted(headcount):
        total = headcount[dept]
        p = present.get(dept, 0)
        report.append({
            'department': dept,
            'total_employees': total,
            'present_today': p,
            'attendance_rate': f"{(p / total * 100) if total else 0:.1f}%",
            'rate_value': round((p / total * 100) if total else 0, 1),
            'late_arrivals': late.get(dept, 0),
        })
    return report


@app.route('/reports')
@login_required
def reports():
    day = resolve_report_day(request.args.get('date'))
    return render_template('reports.html', reports=build_department_report(day),
                           report_day=day, today=today_local())


@app.route('/export/pdf')
@login_required
def export_pdf():
    day = resolve_report_day(request.args.get('date'))
    rendered = render_template('report_template.html', reports=build_department_report(day), report_day=day)
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(rendered, dest=pdf)
    if pisa_status.err:
        return "Error generating PDF", 500
    pdf.seek(0)
    return send_file(pdf, download_name=f"attendance_report_{day}.pdf", as_attachment=True)


@app.route('/export/excel')
@login_required
def export_excel():
    day = resolve_report_day(request.args.get('date'))
    df = pd.DataFrame([{
        'Department': r['department'],
        'Total Employees': r['total_employees'],
        'Present': r['present_today'],
        'Attendance Rate (%)': r['rate_value'],
        'Late Arrivals': r['late_arrivals'],
    } for r in build_department_report(day)])
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=f'Attendance {day}')
    output.seek(0)
    return send_file(output, download_name=f'attendance_report_{day}.xlsx', as_attachment=True)


# ------------------ Leave Management ------------------

@app.route('/leave_management')
@login_required
def leave_management():
    status_filter = request.args.get('status')
    query = Leave.query.join(Employee)
    if status_filter:
        query = query.filter(Leave.status == status_filter)
    leaves = query.all()
    return render_template('leave_management.html', leaves=leaves)


@app.route('/holiday/add', methods=['POST'])
@login_required
def add_holiday():
    name = request.form['name']
    holiday_date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
    flash(f"Holiday '{name}' on {holiday_date} added!")
    return redirect(url_for('leave_management'))


@app.route('/leave/request', methods=['POST'])
@login_required
def request_leave():
    from_date = datetime.strptime(request.form['from_date'], '%Y-%m-%d').date()
    to_date = datetime.strptime(request.form['to_date'], '%Y-%m-%d').date()
    if to_date < from_date:
        flash('Leave end date cannot be before the start date.')
        return redirect(url_for('leave_management'))
    new_leave = Leave(
        employee_id=request.form['employee_id'],
        leave_type=request.form['leave_type'],
        from_date=from_date,
        to_date=to_date,
        reason=request.form['reason']
    )
    db.session.add(new_leave)
    db.session.commit()
    return redirect(url_for('leave_management'))


@app.route('/leave/approve/<int:leave_id>')
@login_required
def approve_leave(leave_id):
    leave = db.get_or_404(Leave, leave_id)
    leave.status = 'Approved'
    db.session.commit()
    return redirect(url_for('leave_management'))


@app.route('/leave/reject/<int:leave_id>')
@login_required
def reject_leave(leave_id):
    leave = db.get_or_404(Leave, leave_id)
    leave.status = 'Rejected'
    db.session.commit()
    return redirect(url_for('leave_management'))


# ------------------ System Settings ------------------

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    settings_row = SystemSettings.query.first()
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        timezone = request.form.get('timezone')
        confidence_threshold = request.form.get('confidence_threshold')

        if settings_row:
            settings_row.company_name = company_name
            settings_row.timezone = timezone
            if confidence_threshold:
                try:
                    value = float(confidence_threshold)
                except ValueError:
                    value = None
                # the UI may send a percentage (55) or a fraction (0.55)
                if value is not None and value > 1:
                    value /= 100
                if value is None or not 0.3 <= value <= 0.9:
                    flash('Confidence threshold must be between 30% and 90%.')
                    return redirect(url_for('settings'))
                settings_row.confidence_threshold = value
        else:
            settings_row = SystemSettings(company_name=company_name, timezone=timezone)
            db.session.add(settings_row)

        db.session.commit()
        flash("Settings updated successfully.")
        return redirect(url_for('settings'))

    return render_template('settings.html', settings=settings_row)


# ------------------ App Bootstrap ------------------

def init_app():
    with app.app_context():
        db.create_all()
        ensure_admin_user()
        ensure_settings_row()
        ensure_demo_data()


init_demo(app)
init_app()

if __name__ == '__main__':
    app.run(debug=True)