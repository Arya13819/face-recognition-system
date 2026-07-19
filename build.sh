#!/usr/bin/env bash
# Build script for Render (or any host that lets you set a custom build command).
# Installs the app's normal dependencies, then installs dlib-bin (a prebuilt wheel,
# no compiling) and face_recognition/face_recognition_models with --no-deps so pip
# doesn't try to replace it with a from-source `dlib` build.
set -e

pip install --upgrade pip
pip install -r requirements.txt
pip install dlib-bin==20.0.1
pip install --no-deps face_recognition==1.3.0 face_recognition_models==0.3.0
