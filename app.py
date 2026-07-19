import os
import io
import base64
import pickle
from functools import wraps
from datetime import datetime, date

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

# Database config — SQLite by default, override with DATABASE_URL in production
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Match confidence: how close a live face must be to a stored one to count as a match
DEFAULT_CONFIDENCE_THRESHOLD = 0.55

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


# ---------------------- Face Recognition Helpers ----------------------

def compute_face_encoding(image_path):
    """Return the 128-d face encoding for the first face found in an image, or None."""
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None


def load_known_encodings():
    """Pull every stored encoding, paired with its employee, for matching."""
    rows = FaceEncoding.query.all()
    encodings, employees = [], []
    for row in rows:
        try:
            encodings.append(pickle.loads(row.encoding))
            employees.append(row.employee)
        except Exception:
            continue
    return encodings, employees


def decode_base64_image(data_url):
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    img_bytes = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(image)


def mark_attendance_checkin(employee_id):
    """Create today's check-in for an employee if one doesn't exist yet."""
    today = date.today()
    existing = Attendance.query.filter(
        Attendance.employee_id == employee_id,
        func.date(Attendance.check_in) == today
    ).first()
    if existing:
        return False
    record = Attendance(
        employee_id=employee_id,
        check_in=datetime.now(),
        status='Present',
        source='face_recognition'
    )
    db.session.add(record)
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
    present_records = Attendance.query.filter_by(status='Present').count()
    attendance_rate = round((present_records / total_records) * 100, 2) if total_records else 0
    late_arrivals = Attendance.query.filter(
        (Attendance.status == 'Late') |
        (func.strftime('%H:%M', Attendance.check_in) > '09:15')
    ).count()

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
                    db.session.add(FaceEncoding(employee_id=new_employee.id, encoding=pickle.dumps(encoding)))
                    db.session.commit()
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
    employee = Employee.query.get_or_404(id)
    if request.method == 'POST':
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
                    if employee.face_encoding:
                        employee.face_encoding.encoding = pickle.dumps(encoding)
                    else:
                        db.session.add(FaceEncoding(employee_id=employee.id, encoding=pickle.dumps(encoding)))
                    db.session.commit()
                    flash('Photo updated and face re-registered.')
                else:
                    flash('Photo updated, but no face was detected in it.')

        return redirect(url_for('manage_employees'))
    return render_template('edit_employee.html', employee=employee)


@app.route('/employees/delete/<int:id>')
@login_required
def delete_employee(id):
    employee = Employee.query.get_or_404(id)
    db.session.delete(employee)
    db.session.commit()
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
    mark_attendance_checkin(emp_id)
    return redirect(url_for('attendance'))


@app.route('/attendance/check_out/<int:emp_id>')
@login_required
def check_out(emp_id):
    today = datetime.now().date()
    record = Attendance.query.filter_by(employee_id=emp_id).filter(func.date(Attendance.check_in) == today).first()
    if record and not record.check_out:
        record.check_out = datetime.now()
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
    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({'status': 'error', 'message': 'face_recognition library is not installed on this server'}), 503

    data = request.get_json(silent=True) or {}
    image_data = data.get('image')
    camera_name = data.get('camera', 'Default Camera')

    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    settings_row = SystemSettings.query.first()
    threshold = settings_row.confidence_threshold if settings_row else DEFAULT_CONFIDENCE_THRESHOLD

    try:
        frame = decode_base64_image(image_data)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return jsonify({'status': 'no_face', 'message': 'No face detected in frame'})

    unknown_encodings = face_recognition.face_encodings(frame, face_locations)
    known_encodings, known_employees = load_known_encodings()

    best_employee = None
    best_confidence = 0.0

    if known_encodings:
        for unknown_encoding in unknown_encodings:
            distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            best_idx = int(np.argmin(distances))
            confidence = max(0.0, 1.0 - float(distances[best_idx]))
            if confidence > best_confidence:
                best_confidence = confidence
                if confidence >= threshold:
                    best_employee = known_employees[best_idx]

    status = 'Success' if best_employee else 'Unrecognized'
    log = RecognitionLog(
        employee_id=best_employee.id if best_employee else None,
        employee_name=best_employee.name if best_employee else 'Unknown',
        confidence=round(best_confidence * 100, 1),
        camera_name=camera_name,
        status=status
    )
    db.session.add(log)
    db.session.commit()

    checked_in = mark_attendance_checkin(best_employee.id) if best_employee else False

    return jsonify({
        'status': status.lower(),
        'name': best_employee.name if best_employee else 'Unknown',
        'confidence': round(best_confidence * 100, 1),
        'checked_in': checked_in
    })


# ------------------ Reports ------------------

@app.route('/reports')
@login_required
def reports():
    today = datetime.now().date()
    departments = db.session.query(Employee.department).distinct().all()
    department_reports = []

    for dept_tuple in departments:
        dept = dept_tuple[0]
        employees = Employee.query.filter_by(department=dept).all()
        total_employees = len(employees)
        present_today = 0
        late_arrivals = 0

        for emp in employees:
            record = Attendance.query.filter_by(employee_id=emp.id).filter(func.date(Attendance.check_in) == today).first()
            if record:
                present_today += 1
                if record.check_in.time() > datetime.strptime("09:15", "%H:%M").time():
                    late_arrivals += 1

        attendance_rate = (present_today / total_employees) * 100 if total_employees else 0

        department_reports.append({
            'department': dept,
            'total_employees': total_employees,
            'present_today': present_today,
            'attendance_rate': f"{attendance_rate:.1f}%",
            'late_arrivals': late_arrivals
        })

    return render_template('reports.html', reports=department_reports)


def generate_report_data():
    today = date.today()
    departments = db.session.query(Employee.department).distinct().all()

    report_data = []
    for dept_tuple in departments:
        dept = dept_tuple[0]
        total_employees = Employee.query.filter_by(department=dept).count()

        present_today = db.session.query(Attendance).join(Employee).filter(
            Employee.department == dept,
            Attendance.status == 'Present',
            func.date(Attendance.check_in) == today
        ).count()

        late_arrivals = db.session.query(Attendance).join(Employee).filter(
            Employee.department == dept,
            Attendance.status == 'Late',
            func.date(Attendance.check_in) == today
        ).count()

        attendance_rate = (present_today / total_employees) * 100 if total_employees else 0
        report_data.append({
            'department': dept,
            'total_employees': total_employees,
            'present_today': present_today,
            'attendance_rate': f"{attendance_rate:.2f}%",
            'late_arrivals': late_arrivals
        })

    return report_data


@app.route('/export/pdf')
@login_required
def export_pdf():
    report_data = generate_report_data()
    rendered = render_template('report_template.html', reports=report_data)
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(rendered, dest=pdf)
    if pisa_status.err:
        return "Error generating PDF", 500
    pdf.seek(0)
    return send_file(pdf, download_name="attendance_report.pdf", as_attachment=True)


@app.route('/export/excel')
@login_required
def export_excel():
    today = datetime.now().date()
    departments = db.session.query(Employee.department).distinct().all()
    data = []

    for dept_tuple in departments:
        dept = dept_tuple[0]
        employees = Employee.query.filter_by(department=dept).all()
        total = len(employees)
        present = 0
        late = 0

        for emp in employees:
            record = Attendance.query.filter_by(employee_id=emp.id).filter(func.date(Attendance.check_in) == today).first()
            if record:
                present += 1
                if record.check_in.time() > datetime.strptime("09:15", "%H:%M").time():
                    late += 1

        rate = (present / total) * 100 if total else 0
        data.append({
            'Department': dept,
            'Total Employees': total,
            'Present Today': present,
            'Attendance Rate (%)': round(rate, 1),
            'Late Arrivals': late
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance Report')
    output.seek(0)

    return send_file(output, download_name='attendance_report.xlsx', as_attachment=True)


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
    new_leave = Leave(
        employee_id=request.form['employee_id'],
        leave_type=request.form['leave_type'],
        from_date=datetime.strptime(request.form['from_date'], '%Y-%m-%d').date(),
        to_date=datetime.strptime(request.form['to_date'], '%Y-%m-%d').date(),
        reason=request.form['reason']
    )
    db.session.add(new_leave)
    db.session.commit()
    return redirect(url_for('leave_management'))


@app.route('/leave/approve/<int:leave_id>')
@login_required
def approve_leave(leave_id):
    leave = Leave.query.get_or_404(leave_id)
    leave.status = 'Approved'
    db.session.commit()
    return redirect(url_for('leave_management'))


@app.route('/leave/reject/<int:leave_id>')
@login_required
def reject_leave(leave_id):
    leave = Leave.query.get_or_404(leave_id)
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
                settings_row.confidence_threshold = float(confidence_threshold)
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


init_app()

if __name__ == '__main__':
    app.run(debug=True)