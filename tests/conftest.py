"""
Test setup: a throwaway SQLite DB and a fake `face_recognition` module, so the
whole suite runs in seconds on any machine (no dlib / webcam needed).

The fake "encodes" a face as the image's average colour, which makes matching
deterministic: two photos of the same colour match, different colours don't.
"""
import io
import os
import sys
import base64
import tempfile

import numpy as np
import pytest
from PIL import Image

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db")
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "test-pass"
os.environ["SECRET_KEY"] = "test-secret"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


class FakeFaceRecognition:
    @staticmethod
    def load_image_file(path):
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def face_locations(frame):
        h, w = frame.shape[:2]
        if frame.mean() < 5:          # an all-black frame = "no face"
            return []
        return [(h // 4, 3 * w // 4, 3 * h // 4, w // 4)]  # (top, right, bottom, left)

    @staticmethod
    def face_encodings(frame, locations=None):
        vec = np.zeros(128)
        vec[:3] = frame.reshape(-1, 3).mean(axis=0) / 255.0 * 3  # spread colours apart
        return [vec for _ in (locations or [0])]


def solid_png(color, size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    buf.seek(0)
    return buf


def data_url(color, size=(320, 240)):
    return "data:image/png;base64," + base64.b64encode(solid_png(color, size).read()).decode()


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(app_module, "face_recognition", FakeFaceRecognition, raising=False)
    monkeypatch.setattr(app_module, "FACE_RECOGNITION_AVAILABLE", True)
    app_module.app.config["TESTING"] = True
    app_module.encoding_cache.invalidate()
    return app_module


@pytest.fixture
def client(app):
    return app.app.test_client()


@pytest.fixture
def admin(client):
    client.post("/login", data={"username": "admin", "password": "test-pass"})
    return client


@pytest.fixture
def demo(client):
    client.get("/demo")
    return client
