
from flask import Flask, Response, request, jsonify, render_template, redirect, flash, url_for, send_file
from database import db
from models import Employee, Attendance, Leave, SystemSettings
from datetime import datetime,date
import pandas as pd
from io import BytesIO
from xhtml2pdf import pisa
import os
from sqlalchemy import func

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'  # REQUIRED for session-based features like flash

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ---------------------- Routes ----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect('/dashboard')
    return render_template('login.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')


from datetime import datetime, timedelta

@app.route('/dashboard')
def dashboard():
    # Existing stats
    total_employees = Employee.query.count()
    total_leaves = Leave.query.filter_by(status='Approved').count()
    total_records = Attendance.query.count()
    present_records = Attendance.query.filter_by(status='Present').count()
    attendance_rate = round((present_records / total_records) * 100, 2) if total_records else 0
    late_arrivals = Attendance.query.filter(
        (Attendance.status == 'Late') |
        (func.strftime('%H:%M', Attendance.check_in) > '09:15')
    ).count()

    # Recent Attendance (latest 5)
    recent_attendance = Attendance.query.order_by(Attendance.check_in.desc()).limit(5).all()

    # Recent Leave Requests (latest 5)
    recent_leaves = Leave.query.order_by(Leave.from_date.desc()).limit(5).all()

    return render_template(
        'dashboard.html',
        total_employees=total_employees,
        total_leaves=total_leaves,
        attendance_rate=attendance_rate,
        late_arrivals=late_arrivals,
        recent_attendance=recent_attendance,
        recent_leaves=recent_leaves
    )


# ------------------ Employee Management ------------------

@app.route('/employees', methods=['GET', 'POST'])
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
def add_employee():
    if request.method == 'POST':
        date_of_joining = datetime.strptime(request.form['date_of_joining'], '%Y-%m-%d').date()
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
        return redirect('/employees')
    return render_template('add_employee.html')

@app.route('/employees/edit/<int:id>', methods=['GET', 'POST'])
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
        return redirect(url_for('manage_employees'))
    return render_template('edit_employee.html', employee=employee)

@app.route('/employees/delete/<int:id>')
def delete_employee(id):
    employee = Employee.query.get_or_404(id)
    db.session.delete(employee)
    db.session.commit()
    return redirect(url_for('manage_employees'))

# ------------------ Attendance ------------------

@app.route('/attendance')
def attendance():
    employees = Employee.query.all()
    attendance_records = Attendance.query.all()
    return render_template('attendance.html', employees=employees, attendance_records=attendance_records)

@app.route('/attendance/check_in/<int:emp_id>')
def check_in(emp_id):
    today = datetime.now().date()
    existing = Attendance.query.filter_by(employee_id=emp_id).filter(db.func.date(Attendance.check_in) == today).first()
    if not existing:
        new_record = Attendance(
            employee_id=emp_id,
            check_in=datetime.now(),
            status='Present'
        )
        db.session.add(new_record)
        db.session.commit()
    return redirect(url_for('attendance'))

@app.route('/attendance/check_out/<int:emp_id>')
def check_out(emp_id):
    today = datetime.now().date()
    record = Attendance.query.filter_by(employee_id=emp_id).filter(db.func.date(Attendance.check_in) == today).first()
    if record and not record.check_out:
        record.check_out = datetime.now()
        db.session.commit()
    return redirect(url_for('attendance'))

# ------------------ Reports ------------------

@app.route('/reports')
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
            record = Attendance.query.filter_by(employee_id=emp.id).filter(db.func.date(Attendance.check_in) == today).first()
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
            record = Attendance.query.filter_by(employee_id=emp.id).filter(db.func.date(Attendance.check_in) == today).first()
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

# @app.route('/leave_management')
# def leave_management():
#     leaves = Leave.query.join(Employee).all()
#     return render_template('leave_management.html', leaves=leaves)

@app.route('/leave_management')
def leave_management():
    status_filter = request.args.get('status')
    query = Leave.query.join(Employee)
    if status_filter:
        query = query.filter(Leave.status == status_filter)
    leaves = query.all()
    return render_template('leave_management.html', leaves=leaves)

@app.route('/holiday/add', methods=['POST'])
def add_holiday():
    name = request.form['name']
    date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
    flash(f"Holiday '{name}' on {date} added!")  # Optional flash
    # You may want to save this to a Holiday model/table if you have one
    return redirect(url_for('leave_management'))


@app.route('/leave/request', methods=['POST'])
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
def approve_leave(leave_id):
    leave = Leave.query.get_or_404(leave_id)
    leave.status = 'Approved'
    db.session.commit()
    return redirect(url_for('leave_management'))

@app.route('/leave/reject/<int:leave_id>')
def reject_leave(leave_id):
    leave = Leave.query.get_or_404(leave_id)
    leave.status = 'Rejected'
    db.session.commit()
    return redirect(url_for('leave_management'))

# ------------------ System Settings ------------------

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    settings = SystemSettings.query.first()
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        timezone = request.form.get('timezone')

        if settings:
            settings.company_name = company_name
            settings.timezone = timezone
        else:
            settings = SystemSettings(company_name=company_name, timezone=timezone)
            db.session.add(settings)

        db.session.commit()
        flash("Settings updated successfully.")
        return redirect(url_for('settings'))

    return render_template('settings.html', settings=settings)

@app.route('/logout')
def logout():
    return redirect(url_for('index'))

# ------------------ Run the App ------------------

if __name__ == '__main__':
    with app.app_context():
        if not os.path.exists('attendance.db'):
            db.create_all()
    app.run(debug=True)
