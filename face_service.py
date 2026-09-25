"""
Face-matching core for AttendanceAI.

Everything here is pure (numpy only, no Flask / DB), so it is unit-testable
without dlib installed:

- serialize_encoding / deserialize_encoding  — safe storage format (raw float64
  bytes instead of pickle; legacy pickled rows are still readable).
- match_faces   — vectorised nearest-neighbour match of every face in a frame
                  against every enrolled face, in one numpy call.
- EncodingCache — keeps enrolled encodings in memory so a recognition request
                  doesn't re-read and re-decode the whole table on every frame.
"""

import pickle
import threading

import numpy as np

ENCODING_DIM = 128
_RAW_SIZE = ENCODING_DIM * 8  # 128 float64 values


# ---------------------------------------------------------------------------
# Storage format
# ---------------------------------------------------------------------------

def serialize_encoding(encoding):
    """128-d vector -> 1024 raw bytes (float64). Never executes code on load."""
    arr = np.asarray(encoding, dtype=np.float64)
    if arr.shape != (ENCODING_DIM,):
        raise ValueError(f"expected a {ENCODING_DIM}-d encoding, got shape {arr.shape}")
    return arr.tobytes()


def deserialize_encoding(blob):
    """Inverse of serialize_encoding. Falls back to pickle for rows written by v1."""
    if blob is None:
        return None
    if len(blob) == _RAW_SIZE:
        return np.frombuffer(blob, dtype=np.float64).copy()
    # Legacy (v1) rows were pickled numpy arrays written by this app itself.
    arr = np.asarray(pickle.loads(blob), dtype=np.float64)
    return arr if arr.shape == (ENCODING_DIM,) else None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_faces(unknown_encodings, known_matrix, threshold):
    """
    Match every detected face against every enrolled face.

    unknown_encodings : list/array of shape (n_faces, 128)
    known_matrix      : array of shape (n_known, 128)
    threshold         : minimum confidence (0-1) to count as a match

    Returns one dict per detected face:
        {"index": best_known_index or None, "confidence": float 0-1}
    where confidence = 1 - euclidean_distance (clipped to 0).
    """
    unknown = np.asarray(unknown_encodings, dtype=np.float64).reshape(-1, ENCODING_DIM)
    known = np.asarray(known_matrix, dtype=np.float64).reshape(-1, ENCODING_DIM)

    if len(unknown) == 0:
        return []
    if len(known) == 0:
        return [{"index": None, "confidence": 0.0} for _ in range(len(unknown))]

    # (n_faces, n_known) distance matrix in a single vectorised op
    dists = np.linalg.norm(unknown[:, None, :] - known[None, :, :], axis=2)
    best_idx = dists.argmin(axis=1)
    best_conf = np.clip(1.0 - dists[np.arange(len(unknown)), best_idx], 0.0, 1.0)

    results = []
    for idx, conf in zip(best_idx, best_conf):
        conf = float(conf)
        results.append({"index": int(idx) if conf >= threshold else None,
                        "confidence": conf})
    return results


# ---------------------------------------------------------------------------
# In-memory cache of enrolled encodings
# ---------------------------------------------------------------------------

class EncodingCache:
    """
    Holds (employee_ids, names, matrix) for all enrolled faces.
    Call invalidate() whenever an encoding is added/changed/deleted;
    the next get() reloads lazily via the supplied loader.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None

    def invalidate(self):
        with self._lock:
            self._data = None

    def get(self, loader):
        with self._lock:
            if self._data is None:
                ids, names, vectors = loader()
                matrix = (np.vstack(vectors) if vectors
                          else np.empty((0, ENCODING_DIM), dtype=np.float64))
                self._data = (ids, names, matrix)
            return self._data


def downscale_factor(width, max_width=640):
    """How much to shrink a frame before detection (1.0 = no shrink)."""
    if not width or width <= max_width:
        return 1.0
    return max_width / float(width)
