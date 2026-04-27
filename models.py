# from database import db
# from datetime import datetime

# class Employee(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     employee_id = db.Column(db.String(50), unique=True, nullable=False)
#     name = db.Column(db.String(100), nullable=False)
#     email = db.Column(db.String(100), nullable=False)
#     phone = db.Column(db.String(15), nullable=False)
#     department = db.Column(db.String(50), nullable=False)
#     role = db.Column(db.String(50), nullable=False)
#     date_of_joining = db.Column(db.Date, nullable=False)

#     def __repr__(self):
#         return f'<Employee {self.name}>'


# class Attendance(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
#     check_in = db.Column(db.DateTime, default=None)
#     check_out = db.Column(db.DateTime, default=None)
#     status = db.Column(db.String(50), default='Absent')

#     employee = db.relationship('Employee', backref=db.backref('attendances', lazy=True))


from database import db

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    date_of_joining = db.Column(db.Date, nullable=False)

    def __repr__(self):
        return f'<Employee {self.name}>'

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
    check_in = db.Column(db.DateTime, default=None)
    check_out = db.Column(db.DateTime, default=None)
    status = db.Column(db.String(50), default='Absent')

    employee = db.relationship('Employee', backref=db.backref('attendances', lazy=True))

class Leave(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
    leave_type = db.Column(db.String(50))
    from_date = db.Column(db.Date)
    to_date = db.Column(db.Date)
    reason = db.Column(db.String(255))
    status = db.Column(db.String(20), default='Pending')  # Pending / Approved / Rejected

    employee = db.relationship('Employee', backref='leaves')

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100), nullable=False)
    timezone = db.Column(db.String(100), nullable=False)


