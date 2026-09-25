"""
Demo Mode (v1.1) for AttendanceAI.

Gives recruiters a one-click, read-only tour of the app with realistic
pre-seeded data — no credentials, no webcam, no empty dashboards.

Design notes:
- `build_seed_plan()` is PURE (no database, deterministic via a fixed RNG seed),
  so the exact data the demo will contain is unit-testable.
- `apply_seed()` is the thin layer that writes the plan via SQLAlchemy.
- A `before_request` guard makes the demo session read-only: any write
  (POST, or the app's write-style GET routes) is politely blocked.

Wire-up in app.py (2 lines):
    from demo import init_demo, ensure_demo_data
    init_demo(app)                       # after routes are defined
    ensure_demo_data()                   # inside init_app()'s app_context
"""

import os
import secrets
import random
from datetime import datetime, date, time, timedelta

from flask import (
    Blueprint, session, redirect, url_for, flash, request, jsonify
)
from PIL import Image, ImageDraw, ImageFont

DEMO_USERNAME = "demo"
DEMO_ID_PREFIX = "DEMO"

# Write-style GET routes that must be blocked in demo mode
# (everything sent via POST is blocked wholesale).
WRITE_GET_PREFIXES = (
    "/employees/delete",
    "/attendance/check_in",
    "/attendance/check_out",
    "/leave/approve",
    "/leave/reject",
)

demo_bp = Blueprint("demo", __name__)


# ---------------------------------------------------------------------------
# Seed plan (pure + deterministic -> unit-testable)
# ---------------------------------------------------------------------------

DEMO_EMPLOYEES = [
    # (employee_id, name, department, role, avatar_bg)
    ("DEMO001", "Priya Sharma",   "Engineering", "Backend Developer",  "#4b4fcb"),
    ("DEMO002", "Rahul Verma",    "Engineering", "Frontend Developer", "#7c3aed"),
    ("DEMO003", "Ananya Iyer",    "Engineering", "QA Engineer",        "#0ea5e9"),
    ("DEMO004", "Vikram Singh",   "Sales",       "Sales Executive",    "#f59e0b"),
    ("DEMO005", "Sneha Patel",    "Sales",       "Account Manager",    "#ef4444"),
    ("DEMO006", "Arjun Mehta",    "HR",          "HR Executive",       "#10b981"),
    ("DEMO007", "Kavya Reddy",    "HR",          "Recruiter",          "#ec4899"),
    ("DEMO008", "Rohan Das",      "Operations",  "Ops Analyst",        "#8b5cf6"),
    ("DEMO009", "Ishita Bansal",  "Operations",  "Logistics Lead",     "#14b8a6"),
]

LEAVE_SEED = [
    # (employee_idx, leave_type, days_from_today_start, length_days, reason, status)
    (1, "Sick Leave",   -6, 1, "Fever and rest advised by doctor", "Approved"),
    (4, "Casual Leave",  3, 2, "Family function out of town",      "Pending"),
    (6, "Casual Leave",  7, 1, "Personal errand",                  "Pending"),
    (8, "Earned Leave", -12, 3, "Planned vacation",                "Rejected"),
]


def build_seed_plan(today=None):
    """Return the full demo dataset as plain dicts. Pure and deterministic."""
    today = today or date.today()
    rng = random.Random(42)  # fixed seed => same demo data every time

    employees = []
    for emp_id, name, dept, role, color in DEMO_EMPLOYEES:
        employees.append({
            "employee_id": emp_id,
            "name": name,
            "email": name.lower().replace(" ", ".") + "@demo.attendanceai.app",
            "phone": "98" + "".join(str(rng.randint(0, 9)) for _ in range(8)),
            "department": dept,
            "role": role,
            "date_of_joining": today - timedelta(days=rng.randint(200, 900)),
            "avatar_bg": color,
        })

    # Attendance: last 14 calendar days, weekdays only
    attendance = []
    for day_offset in range(14, 0, -1):
        d = today - timedelta(days=day_offset)
        if d.weekday() >= 5:  # Sat/Sun
            continue
        for idx, emp in enumerate(employees):
            roll = rng.random()
            if roll < 0.08:               # absent (no record)
                continue
            late = roll > 0.80            # ~1 in 5 present days is late
            if late:
                check_in = datetime.combine(d, time(9, rng.randint(20, 45)))
                status = "Late"
            else:
                check_in = datetime.combine(d, time(8, rng.randint(48, 59))) \
                    if rng.random() < 0.5 else datetime.combine(d, time(9, rng.randint(0, 12)))
                status = "Present"
            check_out = datetime.combine(d, time(17, rng.randint(40, 59))) \
                if rng.random() < 0.5 else datetime.combine(d, time(18, rng.randint(0, 25)))
            attendance.append({
                "employee_idx": idx,
                "check_in": check_in,
                "check_out": check_out,
                "status": status,
                "source": "face_recognition" if rng.random() < 0.6 else "manual",
            })

    # Recognition activity feed: last 3 WEEKDAYS, mostly successes + a few unknowns
    recognition_logs = []
    feed_days = []
    probe = 1
    while len(feed_days) < 3 and probe <= 7:
        d = today - timedelta(days=probe)
        if d.weekday() < 5:
            feed_days.append(d)
        probe += 1
    for d in reversed(feed_days):
        for _ in range(rng.randint(8, 12)):
            ts = datetime.combine(d, time(rng.choice([8, 9, 9, 9, 13, 18]),
                                          rng.randint(0, 59), rng.randint(0, 59)))
            if rng.random() < 0.85:
                idx = rng.randrange(len(employees))
                recognition_logs.append({
                    "employee_idx": idx,
                    "employee_name": employees[idx]["name"],
                    "confidence": round(rng.uniform(0.62, 0.94), 2),
                    "status": "Success",
                    "timestamp": ts,
                })
            else:
                recognition_logs.append({
                    "employee_idx": None,
                    "employee_name": "Unknown",
                    "confidence": round(rng.uniform(0.18, 0.48), 2),
                    "status": "Unrecognized",
                    "timestamp": ts,
                })
    recognition_logs.sort(key=lambda r: r["timestamp"])

    leaves = []
    for emp_idx, ltype, start_off, length, reason, status in LEAVE_SEED:
        start = today + timedelta(days=start_off)
        leaves.append({
            "employee_idx": emp_idx,
            "leave_type": ltype,
            "from_date": start,
            "to_date": start + timedelta(days=length - 1),
            "reason": reason,
            "status": status,
        })

    return {
        "employees": employees,
        "attendance": attendance,
        "recognition_logs": recognition_logs,
        "leaves": leaves,
    }


# ---------------------------------------------------------------------------
# Avatar generation (initials on brand-colored tiles — no real faces needed)
# ---------------------------------------------------------------------------

def _initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()


def generate_avatar(name, bg_hex, out_path, size=256):
    img = Image.new("RGB", (size, size), bg_hex)
    draw = ImageDraw.Draw(img)
    text = _initials(name)
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, int(size * 0.42))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]),
              text, fill="white", font=font)
    img.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# Applying the plan (thin SQLAlchemy layer)
# ---------------------------------------------------------------------------

def ensure_demo_data():
    """Idempotent: creates the demo user + seeded dataset if missing.
    Also self-heals avatar files (Render's disk is wiped on redeploys)."""
    from database import db
    from models import Employee, Attendance, Leave, User, RecognitionLog
    from flask import current_app

    upload_folder = current_app.config["UPLOAD_FOLDER"]

    # Demo login account (password is random + unused — login happens via /demo)
    if not User.query.filter_by(username=DEMO_USERNAME).first():
        demo_user = User(username=DEMO_USERNAME, role="viewer")
        demo_user.set_password(secrets.token_urlsafe(24))
        db.session.add(demo_user)
        db.session.commit()

    plan = build_seed_plan()

    existing = {
        e.employee_id: e for e in
        Employee.query.filter(Employee.employee_id.startswith(DEMO_ID_PREFIX)).all()
    }

    if not existing:
        emp_rows = []
        for spec in plan["employees"]:
            avatar_file = f"demo_{spec['employee_id']}.png"
            generate_avatar(spec["name"], spec["avatar_bg"],
                            os.path.join(upload_folder, avatar_file))
            row = Employee(
                employee_id=spec["employee_id"],
                name=spec["name"],
                email=spec["email"],
                phone=spec["phone"],
                department=spec["department"],
                role=spec["role"],
                date_of_joining=spec["date_of_joining"],
                photo_path=f"uploads/{avatar_file}",
            )
            db.session.add(row)
            emp_rows.append(row)
        db.session.flush()  # assign IDs

        for a in plan["attendance"]:
            db.session.add(Attendance(
                employee_id=emp_rows[a["employee_idx"]].id,
                check_in=a["check_in"], check_out=a["check_out"],
                status=a["status"], source=a["source"],
            ))
        for r in plan["recognition_logs"]:
            db.session.add(RecognitionLog(
                employee_id=(emp_rows[r["employee_idx"]].id
                             if r["employee_idx"] is not None else None),
                employee_name=r["employee_name"],
                confidence=r["confidence"],
                status=r["status"], timestamp=r["timestamp"],
            ))
        for l in plan["leaves"]:
            db.session.add(Leave(
                employee_id=emp_rows[l["employee_idx"]].id,
                leave_type=l["leave_type"], from_date=l["from_date"],
                to_date=l["to_date"], reason=l["reason"], status=l["status"],
            ))
        db.session.commit()
    else:
        # Rows exist — just make sure avatar files survived the last redeploy
        for spec in plan["employees"]:
            avatar_file = f"demo_{spec['employee_id']}.png"
            path = os.path.join(upload_folder, avatar_file)
            if not os.path.exists(path):
                generate_avatar(spec["name"], spec["avatar_bg"], path)


# ---------------------------------------------------------------------------
# Demo login + read-only guard
# ---------------------------------------------------------------------------

@demo_bp.route("/demo")
def demo_login():
    """One-click, credential-free demo session (read-only)."""
    from models import User
    user = User.query.filter_by(username=DEMO_USERNAME).first()
    if not user:
        ensure_demo_data()
        user = User.query.filter_by(username=DEMO_USERNAME).first()
    session.clear()
    session["user_id"] = user.id
    session["username"] = "Demo Visitor"
    session["demo"] = True
    flash("You're exploring the live demo — everything is viewable, writes are disabled.")
    return redirect(url_for("dashboard"))


# POST endpoints that are safe in demo mode: recognition runs as a dry run
# (nothing is written) and the visitor's own face lives only in their session.
DEMO_ALLOWED_POSTS = (
    "/api/recognize-face",
    "/api/demo/enroll",
    "/api/demo/reset",
)


def is_write_request(method, path):
    """Pure helper: is this request a write, for demo-guard purposes?"""
    if method == "POST":
        return path not in DEMO_ALLOWED_POSTS
    return any(path.startswith(p) for p in WRITE_GET_PREFIXES)


def init_demo(app):
    app.register_blueprint(demo_bp)

    @app.before_request
    def _demo_read_only_guard():
        if not session.get("demo"):
            return None
        if request.path in ("/logout", "/demo"):
            return None
        if is_write_request(request.method, request.path):
            if request.path.startswith("/api/"):
                return jsonify({
                    "error": "Demo mode is read-only. Clone the repo to try writes!"
                }), 403
            flash("Demo mode is read-only — sign in as admin to make changes.")
            return redirect(request.referrer or url_for("dashboard"))
        return None
