from database import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model):
    """Admin/staff accounts used to log into the dashboard."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='admin')

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)

    def __repr__(self):
        return f'<User {self.username}>'


class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    date_of_joining = db.Column(db.Date, nullable=False)
    photo_path = db.Column(db.String(255), nullable=True)

    face_encoding = db.relationship('FaceEncoding', backref='employee', uselist=False,
                                     cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Employee {self.name}>'


class FaceEncoding(db.Model):
    """128-d face embedding for an employee, generated on photo upload."""
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False, unique=True)
    encoding = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
    check_in = db.Column(db.DateTime, default=None)
    check_out = db.Column(db.DateTime, default=None)
    status = db.Column(db.String(50), default='Absent')
    source = db.Column(db.String(20), default='manual')  # manual / face_recognition

    employee = db.relationship('Employee', backref=db.backref('attendances', lazy=True))


class RecognitionLog(db.Model):
    """Every face-recognition attempt, matched or not, for the activity feed."""
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=True)
    employee_name = db.Column(db.String(100))
    confidence = db.Column(db.Float, default=0.0)
    camera_name = db.Column(db.String(100), default='Default Camera')
    status = db.Column(db.String(20), default='Unrecognized')  # Success / Unrecognized
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    employee = db.relationship('Employee')


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
    confidence_threshold = db.Column(db.Float, default=0.55)
