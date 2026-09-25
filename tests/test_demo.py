from datetime import date

from demo import build_seed_plan, is_write_request


def test_seed_plan_is_deterministic():
    d = date(2026, 3, 16)
    assert build_seed_plan(d) == build_seed_plan(d)


def test_seed_plan_has_no_weekend_attendance():
    plan = build_seed_plan(date(2026, 3, 16))
    assert all(a["check_in"].weekday() < 5 for a in plan["attendance"])


def test_write_guard():
    assert is_write_request("POST", "/employees/add")
    assert is_write_request("GET", "/employees/delete/3")
    assert not is_write_request("GET", "/dashboard")
    assert not is_write_request("POST", "/api/recognize-face")   # dry run in demo
    assert not is_write_request("POST", "/api/demo/enroll")      # session-only
