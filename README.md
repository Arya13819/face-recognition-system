# AttendanceAI — Face Recognition Attendance & Employee Management System

A Flask web app that manages employees, tracks attendance, and can check people in
automatically using real-time face recognition from a webcam — no manual punch-in
required.

## Features

- **Real face recognition** — powered by [`face_recognition`](https://github.com/ageitgey/face_recognition)
  (dlib's ResNet face embeddings). When an employee's photo is uploaded, a 128-dimension
  face encoding is computed and stored. The live recognition page captures webcam frames,
  matches them against every enrolled face, and automatically checks the matched employee
  in for the day.
- **Authenticated dashboard** — session-based login (hashed passwords via Werkzeug),
  every management route requires login.
- **Employee management** — add/edit/delete employees with photo upload and automatic
  face enrollment.
- **Attendance tracking** — manual check-in/check-out plus automatic check-in from
  face recognition, with a `source` field to distinguish the two.
- **Leave management** — request, approve, and reject leave.
- **Reports & analytics** — department-wise attendance rate and late-arrival breakdown,
  exportable as PDF or Excel.
- **Configurable match confidence** — tune how strict the face match needs to be from
  System Settings.

## Tech Stack

| Layer            | Choice                                   |
|-------------------|-------------------------------------------|
| Backend            | Flask, Flask-SQLAlchemy                   |
| Face recognition   | face_recognition (dlib ResNet embeddings) |
| Database           | SQLite (dev) — swap in Postgres for prod  |
| Reports            | pandas, XlsxWriter, xhtml2pdf             |
| Frontend            | Server-rendered Jinja templates, vanilla JS (`getUserMedia` for webcam capture) |
| Deployment          | Gunicorn + Render (or any WSGI host)      |

## How the face recognition actually works

1. **Enrollment** — When an employee is added (or their photo is updated), the photo is
   run through `face_recognition.face_encodings()`, producing a 128-d vector that
   describes their face. That vector is stored in the `FaceEncoding` table — the photo
   itself is never compared directly.
2. **Live matching** — The browser captures a frame from the webcam every few seconds
   and posts it as base64 to `/api/recognize-face`. The server detects any face in the
   frame, computes its encoding, and compares it (Euclidean distance) against every
   enrolled encoding.
3. **Decision** — `confidence = 1 - distance`. If the best match's confidence clears the
   threshold set in System Settings, that employee is checked in for the day (once per
   day) and the attempt is logged either way — matched or not — for auditability.

## Local Setup

```bash
git clone <this-repo-url>
cd Kharcha-Khabhar-... # or wherever you cloned it
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Face recognition deps use a prebuilt dlib wheel to avoid a slow source compile —
# see build.sh for why this is two steps instead of one `pip install -r requirements.txt`
bash build.sh

cp .env.example .env            # then edit SECRET_KEY / ADMIN_PASSWORD
python3 app.py
```

Visit `http://localhost:5000`, log in with the admin credentials from `.env`
(defaults to `admin` / `admin123` if unset — **change this before deploying**).

## Deployment (Render)

1. Push this repo to GitHub (public or private — Render supports both).
2. On Render: **New → Web Service**, connect the repo.
3. **Build Command:** `bash build.sh`
4. **Start Command:** `gunicorn app:app`
5. Add environment variables: `SECRET_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and
   optionally `DATABASE_URL` (see below).

### ⚠️ Important: persistent storage

Render's free/starter web services use an **ephemeral filesystem** — anything written
to disk (the SQLite database, uploaded employee photos) is wiped on every redeploy or
restart. For a demo this is often fine, but for anything real:

- Add a [Render Persistent Disk](https://render.com/docs/disks) mounted at this
  project's folder, **or**
- Point `DATABASE_URL` at a managed Postgres instance (Render's free Postgres works)
  and move uploaded photos to an object store (e.g. Cloudinary, S3) instead of local disk.

## Why the build is a two-step script

`face_recognition` depends on `dlib`, and installing plain `dlib` from PyPI compiles
it from C++ source — this regularly takes 10+ minutes and times out on most free CI/
build environments. `build.sh` installs the prebuilt `dlib-bin` wheel instead, then
installs `face_recognition`/`face_recognition_models` with `--no-deps` so pip doesn't
try to replace it with a from-source build. Build time drops from 10+ minutes to under
a minute.

## Project Structure

```
.
├── app.py                  # Routes, auth, face recognition matching logic
├── models.py                # SQLAlchemy models (User, Employee, FaceEncoding, Attendance, ...)
├── database.py               # SQLAlchemy instance
├── requirements.txt           # Core dependencies
├── build.sh                   # Handles the dlib-bin / face_recognition install
├── Procfile                    # gunicorn start command for Render/Heroku-style hosts
├── templates/                  # Jinja2 templates (dashboard, employees, attendance, ...)
└── static/uploads/               # Employee photos
```
