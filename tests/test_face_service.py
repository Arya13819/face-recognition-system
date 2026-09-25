import pickle

import numpy as np
import pytest

from face_service import (
    serialize_encoding, deserialize_encoding, match_faces, EncodingCache, downscale_factor
)


def vec(*head):
    v = np.zeros(128)
    v[:len(head)] = head
    return v


def test_serialize_round_trip():
    v = np.random.default_rng(0).normal(size=128)
    blob = serialize_encoding(v)
    assert len(blob) == 1024
    assert np.allclose(deserialize_encoding(blob), v)


def test_legacy_pickled_rows_still_load():
    v = np.arange(128, dtype=float)
    assert np.allclose(deserialize_encoding(pickle.dumps(v)), v)


def test_serialize_rejects_wrong_shape():
    with pytest.raises(ValueError):
        serialize_encoding(np.zeros(64))


def test_match_above_and_below_threshold():
    known = np.vstack([vec(1.0), vec(0, 1.0)])
    res = match_faces([vec(0.95), vec(0.5, 0.5)], known, threshold=0.6)
    assert res[0]["index"] == 0 and res[0]["confidence"] == pytest.approx(0.95)
    assert res[1]["index"] is None          # equidistant & too far from both


def test_match_every_face_in_frame():
    known = np.vstack([vec(1.0), vec(0, 1.0)])
    res = match_faces([vec(0, 1.0), vec(1.0)], known, threshold=0.5)
    assert [r["index"] for r in res] == [1, 0]


def test_match_with_nobody_enrolled():
    res = match_faces([vec(1.0)], np.empty((0, 128)), threshold=0.5)
    assert res == [{"index": None, "confidence": 0.0}]


def test_cache_loads_once_until_invalidated():
    calls = []

    def loader():
        calls.append(1)
        return [7], ["Asha"], [vec(1.0)]

    cache = EncodingCache()
    cache.get(loader)
    cache.get(loader)
    assert len(calls) == 1
    cache.invalidate()
    ids, names, matrix = cache.get(loader)
    assert len(calls) == 2 and ids == [7] and matrix.shape == (1, 128)


@pytest.mark.parametrize("width,expected", [(320, 1.0), (640, 1.0), (1280, 0.5), (None, 1.0)])
def test_downscale_factor(width, expected):
    assert downscale_factor(width) == expected
