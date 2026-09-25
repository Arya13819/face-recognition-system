# AttendanceAI — Face Recognition Attendance & Employee Management System

[![tests](https://github.com/Arya13819/face-recognition-system/actions/workflows/tests.yml/badge.svg)](https://github.com/Arya13819/face-recognition-system/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Flask](https://img.shields.io/badge/flask-3.0-lightgrey)
![face_recognition](https://img.shields.io/badge/dlib-ResNet%20embeddings-orange)

A Flask web app that manages employees, tracks attendance, and checks people in
automatically using real-time face recognition from a webcam — no manual punch-in
required.

### ▶️ Try it live — no login needed

**[Open the live demo →](https://face-recognition-system-0z08.onrender.com/demo)**

1. Go to **Face Recognition** → **Start Camera**
2. Type your name → **Enroll my face**
3. Move around — the app draws a box around your face and recognises you live,
   showing server processing time for every frame.

The demo is a dry run: your face encoding lives only in your browser session and
nothing is written to the database.

<!-- Add a 20-second GIF here: docs/demo.gif -->

## Architecture

```mermaid
flowchart LR
    A[Browser webcam<br/>getUserMedia] -->|JPEG frame, 640px<br/>every 1.5 s| B[/api/recognize-face/]
    B --> C[Detect faces<br/>HOG, dlib]
    C --> D[128-d embeddings<br/>dlib ResNet]
    D --> E{Vectorised match<br/>vs in-memory cache}
    E -->|confidence >= threshold| F[Check in once/day<br/>Present / Late]
    E --> G[RecognitionLog<br/>matched or not]
    F --> H[(SQLite / Postgres)]
    G --> H
    H --> I[Dashboard · Reports · PDF / Excel]
```

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
- **Multi-face frames** — every face in the frame is matched and checked in, not just one.
- **Timezone-aware** — attendance is recorded in the company's timezone (default
  `Asia/Kolkata`), so a UTC server still marks a 09:05 IST check-in correctly and
  flags check-ins after `LATE_CUTOFF` (default 09:15) as **Late**.
- **Recruiter demo** — one-click read-only tour with seeded data, plus "enroll your own
  face" so visitors can test recognition on themselves.

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

### Engineering notes

- **Fast matching** — enrolled encodings are cached in memory and matched with a single
  vectorised numpy distance matrix (`face_service.match_faces`), instead of re-reading
  and unpickling the whole table on every frame. The cache is invalidated whenever a
  photo is added, changed or deleted.
- **Smaller frames** — the browser downscales frames to 640 px before upload and the
  server downscales anything larger, since HOG detection time grows with pixel count.
- **Safe storage** — encodings are stored as raw `float64` bytes, not pickle, so loading
  them can never execute code (rows from v1 are still readable).
- **No request pile-up** — the client never sends a new frame while one is in flight.

## Results

End-to-end check with the real `face_recognition` library, using the sample photos
from the `face_recognition` project (2 people enrolled, one photo each):

| Probe                          | Result                                    |
|--------------------------------|-------------------------------------------|
| Another photo of an enrolled person | ✅ matched                           |
| Frame with two enrolled people | ✅ both matched & checked in              |
| Stranger                        | ✅ rejected as Unknown                    |
| Same person, second frame       | ✅ not double-checked-in                  |

Server time per frame on a small shared CPU: roughly 230–580 ms (HOG detection +
encoding + match). To benchmark on a real dataset such as LFW:

```bash
python evaluate.py path/to/dataset --threshold 0.55 --impostors 20
```

It prints identification accuracy, false-accept / false-reject rates and p50/p95
latency.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

31 tests cover encoding storage, matching, the once-per-day rule, late arrivals,
report consistency, validation, and demo-mode guarantees (dry run, no writes). They use
a fake `face_recognition` module, so they run in about 2 seconds without dlib and
run on every push via GitHub Actions.

## Local Setup

```bash
git clone https://github.com/Arya13819/face-recognition-system.git
cd face-recognition-system
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
   optionally `DATABASE_URL`, `LATE_CUTOFF` (default `09:15`).
6. **Health check path:** `/healthz`

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
├── app.py                  # Routes, auth, attendance rules, recognition endpoint
├── face_service.py          # Pure matching core: storage format, vectorised match, cache
├── demo.py                  # Recruiter demo: seeded data, read-only guard, try-it-yourself
├── evaluate.py              # Accuracy / latency benchmark on a labelled photo folder
├── models.py                # SQLAlchemy models (User, Employee, FaceEncoding, Attendance, ...)
├── database.py               # SQLAlchemy instance
├── requirements.txt           # Core dependencies
├── build.sh                   # Handles the dlib-bin / face_recognition install
├── Procfile                    # gunicorn start command for Render/Heroku-style hosts
├── tests/                      # pytest suite (fake face_recognition, runs in ~2 s)
├── .github/workflows/          # CI: runs the tests on every push
├── templates/                  # Jinja2 templates (dashboard, employees, attendance, ...)
└── static/uploads/               # Employee photos
```
