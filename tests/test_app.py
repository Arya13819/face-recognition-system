from datetime import datetime, date, time

from models import Attendance, Employee, RecognitionLog, FaceEncoding
from conftest import solid_png, data_url


def add_employee(client, emp_id, color, name=None, dept="QA"):
    return client.post("/employees/add", data={
        "employee_id": emp_id, "name": name or f"Emp {emp_id}", "email": f"{emp_id}@x.com",
        "phone": "9999999999", "department": dept, "role": "Tester",
        "date_of_joining": "2025-01-01",
        "photo": (solid_png(color), "face.png"),
    }, content_type="multipart/form-data")


def employee(app, emp_id):
    with app.app.app_context():
        return Employee.query.filter_by(employee_id=emp_id).first()


# ---------------- auth ----------------

def test_pages_require_login(client):
    for path in ["/dashboard", "/employees", "/reports", "/face_recognition", "/settings"]:
        assert client.get(path).status_code == 302


def test_wrong_password_rejected(client):
    r = client.post("/login", data={"username": "admin", "password": "nope"})
    assert b"Invalid username or password" in r.data


def test_all_pages_render_for_admin(admin):
    for path in ["/dashboard", "/employees", "/attendance", "/face_recognition",
                 "/reports", "/leave_management", "/settings", "/export/pdf", "/export/excel"]:
        assert admin.get(path).status_code == 200, path


def test_healthz(client):
    assert client.get("/healthz").get_json()["status"] == "ok"


# ---------------- attendance rules ----------------

def test_late_status_uses_cutoff(app):
    assert app.status_for_checkin(datetime(2026, 1, 5, 9, 10)) == "Present"
    assert app.status_for_checkin(datetime(2026, 1, 5, 9, 16)) == "Late"


def test_checkin_only_once_per_day_in_local_time(app, admin, monkeypatch):
    add_employee(admin, "T-ONCE", (10, 200, 10))
    emp = employee(app, "T-ONCE")
    monkeypatch.setattr(app, "now_local", lambda: datetime(2026, 3, 2, 9, 40))
    with app.app.app_context():
        assert app.mark_attendance_checkin(emp.id) is True
        assert app.mark_attendance_checkin(emp.id) is False
        rec = Attendance.query.filter_by(employee_id=emp.id).one()
        assert rec.status == "Late" and rec.check_in == datetime(2026, 3, 2, 9, 40)


def test_report_page_and_exports_agree_on_late_count(app, admin, monkeypatch):
    add_employee(admin, "T-RPT", (200, 200, 10), dept="ReportDept")
    emp = employee(app, "T-RPT")
    monkeypatch.setattr(app, "now_local", lambda: datetime(2026, 3, 3, 9, 50))
    with app.app.app_context():
        app.mark_attendance_checkin(emp.id)
        row = [r for r in app.build_department_report(date(2026, 3, 3)) if r["department"] == "ReportDept"][0]
    assert row["present_today"] == 1 and row["late_arrivals"] == 1
    assert admin.get("/export/pdf?date=2026-03-03").status_code == 200
    assert admin.get("/export/excel?date=2026-03-03").status_code == 200


def test_edit_rejects_duplicate_employee_id(app, admin):
    add_employee(admin, "T-A", (1, 2, 200))
    add_employee(admin, "T-B", (1, 200, 2))
    b = employee(app, "T-B")
    r = admin.post(f"/employees/edit/{b.id}", data={
        "employee_id": "T-A", "name": "B", "email": "b@x.com", "phone": "1",
        "department": "QA", "role": "R", "date_of_joining": "2025-01-01"})
    assert b"belongs to" in r.data
    assert employee(app, "T-B") is not None


def test_leave_end_before_start_is_rejected(app, admin):
    add_employee(admin, "T-LV", (90, 90, 90))
    emp = employee(app, "T-LV")
    r = admin.post("/leave/request", data={"employee_id": emp.id, "leave_type": "Casual",
                                          "from_date": "2026-05-10", "to_date": "2026-05-01",
                                          "reason": "x"}, follow_redirects=True)
    assert b"cannot be before" in r.data


# ---------------- face recognition ----------------

def test_enrolment_stores_safe_encoding(app, admin):
    add_employee(admin, "T-ENC", (250, 10, 10))
    emp = employee(app, "T-ENC")
    with app.app.app_context():
        assert len(FaceEncoding.query.filter_by(employee_id=emp.id).one().encoding) == 1024


def test_recognition_checks_in_and_logs(app, admin):
    add_employee(admin, "T-REC", (10, 10, 250), name="Blue Person")
    r = admin.post("/api/recognize-face", json={"image": data_url((10, 10, 250))}).get_json()
    assert r["status"] == "success" and r["name"] == "Blue Person"
    assert r["faces"][0]["box"]["right"] > r["faces"][0]["box"]["left"]
    assert "processing_ms" in r
    emp = employee(app, "T-REC")
    with app.app.app_context():
        assert Attendance.query.filter_by(employee_id=emp.id).count() == 1
        assert RecognitionLog.query.filter_by(employee_id=emp.id).count() == 1


def test_unknown_face_and_no_face(admin):
    r = admin.post("/api/recognize-face", json={"image": data_url((123, 77, 5))}).get_json()
    assert r["status"] == "unrecognized"
    r = admin.post("/api/recognize-face", json={"image": data_url((0, 0, 0))}).get_json()
    assert r["status"] == "no_face"


def test_bad_payloads(admin):
    assert admin.post("/api/recognize-face", json={}).status_code == 400
    assert admin.post("/api/recognize-face", json={"image": "not-base64!!"}).status_code == 400


# ---------------- demo mode ----------------

def test_demo_recognition_is_a_dry_run(app, demo):
    with app.app.app_context():
        logs, att = RecognitionLog.query.count(), Attendance.query.count()
    r = demo.post("/api/recognize-face", json={"image": data_url((40, 160, 40))}).get_json()
    assert r["demo"] is True
    with app.app.app_context():
        assert RecognitionLog.query.count() == logs and Attendance.query.count() == att


def test_demo_visitor_can_enroll_and_be_recognized(demo):
    r = demo.post("/api/demo/enroll", json={"image": data_url((180, 60, 200)), "name": "Recruiter"})
    assert r.get_json()["status"] == "enrolled"
    r = demo.post("/api/recognize-face", json={"image": data_url((180, 60, 200))}).get_json()
    assert r["status"] == "success" and r["name"] == "Recruiter"
    demo.post("/api/demo/reset")
    r = demo.post("/api/recognize-face", json={"image": data_url((180, 60, 200))}).get_json()
    assert r["name"] != "Recruiter"


def test_demo_still_blocks_real_writes(app, demo):
    with app.app.app_context():
        before = Employee.query.count()
    demo.post("/employees/add", data={"employee_id": "HACK"})
    with app.app.app_context():
        assert Employee.query.count() == before


def test_demo_enroll_not_available_to_admin(admin):
    assert admin.post("/api/demo/enroll", json={"image": data_url((1, 1, 1))}).status_code == 403
