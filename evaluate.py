"""
Benchmark AttendanceAI's matcher on a labelled photo folder.

Backs up the accuracy / latency numbers in the README with a reproducible run.

Dataset layout (one sub-folder per person, 2+ photos each):

    dataset/
      alice/ 1.jpg 2.jpg 3.jpg
      bob/   1.jpg 2.jpg
      ...

Protocol:
  * For each person, the FIRST photo is enrolled (like uploading an employee photo);
    every other photo is a probe (like a webcam frame).
  * --impostors N : the last N people are NOT enrolled, so their probes must come
    back "Unknown" - this measures false accepts.

Usage:
    python evaluate.py dataset/ --threshold 0.55 --impostors 20

A good public dataset is LFW (Labeled Faces in the Wild); keep people with >= 2 photos.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from face_service import match_faces, downscale_factor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode(face_recognition, path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    scale = downscale_factor(img.width)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    frame = np.array(img)
    t0 = time.perf_counter()
    locations = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, locations[:1]) if locations else []
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return (encodings[0] if encodings else None), elapsed_ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--impostors", type=int, default=0, help="people held out of enrolment")
    ap.add_argument("--max-people", type=int, default=0, help="limit for a quick run (0 = all)")
    args = ap.parse_args()

    try:
        import face_recognition
    except ImportError:
        sys.exit("face_recognition is not installed - run `bash build.sh` first.")

    people = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    people = [p for p in people if len([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS]) >= 2]
    if args.max_people:
        people = people[:args.max_people]
    if len(people) <= args.impostors:
        sys.exit("Need more people than --impostors.")
    enrolled_people, impostor_people = people[:len(people) - args.impostors], people[len(people) - args.impostors:]

    names, vectors, latencies = [], [], []
    probes = []  # (true_name or None, path)
    for person in people:
        files = sorted(f for f in person.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if person in enrolled_people:
            vec, ms = encode(face_recognition, files[0])
            if vec is None:
                continue
            names.append(person.name)
            vectors.append(vec)
            probes += [(person.name, f) for f in files[1:]]
        else:
            probes += [(None, f) for f in files]

    known = np.vstack(vectors)
    correct = wrong_person = missed = false_accept = true_reject = no_face = 0
    genuine = impostor = 0
    for true_name, path in probes:
        vec, ms = encode(face_recognition, path)
        if vec is None:
            no_face += 1
            continue
        t0 = time.perf_counter()
        result = match_faces([vec], known, args.threshold)[0]
        latencies.append(ms + (time.perf_counter() - t0) * 1000)
        predicted = names[result["index"]] if result["index"] is not None else None
        if true_name is not None:
            genuine += 1
            if predicted == true_name:
                correct += 1
            elif predicted is None:
                missed += 1
            else:
                wrong_person += 1
        else:
            impostor += 1
            if predicted is None:
                true_reject += 1
            else:
                false_accept += 1

    pct = lambda a, b: f"{(100 * a / b):.1f}%" if b else "n/a"
    lat = sorted(latencies)
    print(f"\nAttendanceAI benchmark  (threshold={args.threshold})")
    print(f"  enrolled people        : {len(names)}")
    print(f"  genuine probes         : {genuine}")
    print(f"  impostor probes        : {impostor}")
    print(f"  skipped (no face found): {no_face}")
    print(f"  identification accuracy: {pct(correct, genuine)}   (right person, among enrolled)")
    print(f"  false rejects          : {pct(missed, genuine)}")
    print(f"  wrong person           : {pct(wrong_person, genuine)}")
    print(f"  false accepts          : {pct(false_accept, impostor)}   (stranger matched to someone)")
    if lat:
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        print(f"  latency / frame        : p50 {statistics.median(lat):.0f} ms, p95 {p95:.0f} ms (detect + encode + match)")


if __name__ == "__main__":
    main()
