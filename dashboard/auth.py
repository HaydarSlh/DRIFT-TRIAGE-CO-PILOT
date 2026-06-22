"""Lightweight account system for the dashboard.

Why local SQLite + scrypt: the dashboard isn't on the Postgres instance and we
don't want to add a new service for what is essentially a five-row table. The
DB file lives under ``/app/data/dashboard.db`` inside the container; mount it
as a volume in compose if you want auth to survive restarts.

Roles:
  - viewer    — read-only access (metrics, lists)
  - reviewer  — viewer + can approve/reject HIL items
  - admin     — reviewer + can promote/rollback registry versions, requeue DLQ

The first account that signs up gets ``admin`` automatically (bootstrap).
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st

DB_PATH = Path(os.environ.get("DASHBOARD_DB_PATH", "/app/data/dashboard.db"))
ROLES = ("viewer", "reviewer", "admin")
ROLE_RANK = {"viewer": 0, "reviewer": 1, "admin": 2}


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username     TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                role          TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
            """
        )


@contextmanager
def _conn():
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Password hashing — scrypt is in stdlib and resists GPU brute force
# ---------------------------------------------------------------------------


def _hash(password: str, salt: str) -> str:
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return derived.hex()


def _verify(password: str, salt: str, expected: str) -> bool:
    return secrets.compare_digest(_hash(password, salt), expected)


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------


def user_count() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])


def create_user(username: str, password: str, role: str = "viewer") -> tuple[bool, str]:
    """Returns (ok, message)."""
    username = username.strip().lower()
    if len(username) < 3 or not username.replace("_", "").isalnum():
        return False, "Username must be ≥3 chars, alphanumeric or underscores."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if role not in ROLES:
        return False, f"Invalid role. Choose from: {', '.join(ROLES)}."

    if user_count() == 0:
        role = "admin"  # bootstrap

    salt = secrets.token_hex(16)
    pw_hash = _hash(password, salt)
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, pw_hash, salt, role, datetime.utcnow().isoformat()),
            )
            c.commit()
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    return True, f"Account created (role: {role})."


def authenticate(username: str, password: str) -> dict | None:
    username = username.strip().lower()
    with _conn() as c:
        row = c.execute(
            "SELECT username, password_hash, salt, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return None
    if not _verify(password, row["salt"], row["password_hash"]):
        return None
    return {"username": row["username"], "role": row["role"]}


# ---------------------------------------------------------------------------
# Streamlit-side helpers
# ---------------------------------------------------------------------------


def current_user() -> dict | None:
    return st.session_state.get("user")


def is_authed() -> bool:
    return current_user() is not None


def has_role(min_role: str) -> bool:
    user = current_user()
    if user is None:
        return False
    return ROLE_RANK.get(user["role"], -1) >= ROLE_RANK[min_role]


def require_role(min_role: str) -> bool:
    if not has_role(min_role):
        st.error(f"This action requires the **{min_role}** role.")
        return False
    return True


def logout() -> None:
    st.session_state.pop("user", None)


def render_login_signup() -> None:
    """Render the login/signup gate. Called from app.py before any page."""
    st.markdown("# Drift Triage Co-Pilot")
    st.caption("Sign in to access the dashboard.")

    bootstrap = user_count() == 0
    if bootstrap:
        st.info("First-time setup — the next account you create will be the **admin**.")

    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pw")
            ok = st.form_submit_button("Sign in", use_container_width=True, type="primary")
            if ok:
                user = authenticate(u, p)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            u = st.text_input("Username", key="signup_user")
            p = st.text_input("Password (min 8 chars)", type="password", key="signup_pw")
            p2 = st.text_input("Confirm password", type="password", key="signup_pw2")
            requested_role = st.selectbox(
                "Requested role",
                ROLES,
                index=0,
                help="Bootstrapping the first account always grants admin.",
            )
            ok = st.form_submit_button("Create account", use_container_width=True)
            if ok:
                if p != p2:
                    st.error("Passwords don't match.")
                else:
                    success, msg = create_user(u, p, requested_role)
                    (st.success if success else st.error)(msg)
