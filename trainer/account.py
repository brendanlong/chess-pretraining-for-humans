"""Operator access to user rows, from the server's shell.

The app covers anyone who can sign in: signup claims the guest row you are
currently playing on, and the settings drawer deletes the account you are
signed into. This is for the rows the app can't reach — the ones that predate
accounts (the old `?user=name` profiles), a password nobody remembers, and an
emailed deletion request from the address on the account:

    uv run python -m trainer.account list
    uv run python -m trainer.account set-password brendan
    uv run python -m trainer.account delete brendan
"""

import argparse
import getpass
import sqlite3
import sys
from pathlib import Path

from . import auth
from .db import DEFAULT_DB, connect


def set_password(conn: sqlite3.Connection, user: dict, rename_to: str | None) -> int:
    try:
        name = auth.check_username(rename_to or user["name"])
        if name.lower() != user["name"].lower() and auth.find_by_username(conn, name):
            raise auth.AuthError(f"username {name!r} is taken")
        password = getpass.getpass("New password: ")
        if password != getpass.getpass("Repeat: "):
            raise auth.AuthError("passwords don't match")
        auth.check_password(password)
    except auth.AuthError as e:
        print(e, file=sys.stderr)
        return 1
    conn.execute(
        "UPDATE users SET name = ?, password_hash = ? WHERE id = ?",
        (name, auth.hash_password(password), user["id"]),
    )
    conn.commit()
    # This command is the only recovery path the app has — there is no in-app
    # password change and no reset email yet — so it has to assume the reason
    # it's being run is that someone else knows the old password. Rotating the
    # hash while leaving their session live would recover nothing.
    revoked = auth.revoke_sessions(conn, user["id"])
    print(
        f"{name} can now sign in ({user['attempts']} trials preserved"
        + (f", {revoked} existing session(s) signed out)" if revoked else ")")
    )
    return 0


def delete(conn: sqlite3.Connection, user: dict, assume_yes: bool) -> int:
    """Erase a user and everything attached to them.

    This destroys responses, which is the experimental record — so say what is
    about to go and make confirming it deliberate. Typing the name back is the
    guard: `delete brendan` a line above `delete brendan-old` in a terminal is
    exactly how the wrong row gets erased.
    """
    responses = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE user_id = ?", (user["id"],)
    ).fetchone()[0]
    kind = "guest" if auth.is_guest(user) else "account"
    print(
        f"{user['name']} ({kind}): {responses} responses, {user['attempts']} trials, "
        f"email {user['email'] or 'none'}"
    )
    if not assume_yes:
        typed = input("Erase all of it? Type the username to confirm: ").strip()
        if typed.lower() != user["name"].lower():
            print("not confirmed; nothing deleted", file=sys.stderr)
            return 1
    counts = auth.delete_user(conn, user["id"])
    print(
        f"deleted {user['name']}: {counts['responses']} responses, "
        f"{counts['sessions']} sessions, {counts['users']} user row"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show users, trial counts, and whether they have a password")
    sp = sub.add_parser("set-password", help="give a user a password so they can sign in")
    sp.add_argument("name")
    sp.add_argument("--rename-to", help="also give the row a login-friendly username")
    dp = sub.add_parser("delete", help="erase a user, their sessions, and all their responses")
    dp.add_argument("name")
    dp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    conn = connect(args.db)

    if args.cmd == "list":
        for row in conn.execute(
            "SELECT name, attempts, password_hash IS NOT NULL AS claimed FROM users ORDER BY id"
        ):
            kind = "account" if row["claimed"] else "guest"
            print(f"{row['name']:<40} {row['attempts']:>6} trials  {kind}")
        return 0

    try:
        user = auth.find_by_username(conn, args.name)
    except auth.AuthError as e:
        # An ambiguous name, on a database missing the case-insensitive unique
        # index. Refusing is the point: acting on a guess here is how one
        # user's password ends up on another user's row.
        print(e, file=sys.stderr)
        return 1
    if user is None:
        print(f"no user named {args.name!r}", file=sys.stderr)
        return 1

    if args.cmd == "delete":
        return delete(conn, user, args.yes)
    return set_password(conn, user, args.rename_to)


if __name__ == "__main__":
    raise SystemExit(main())
