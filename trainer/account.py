"""Set a password on an existing user row, from the server's shell.

Signup in the app claims the guest row you are currently playing on, which
leaves nothing to attach to for rows that predate accounts (the old
`?user=name` profiles) or for a password nobody remembers. This is the
operator's way in:

    uv run python -m trainer.account list
    uv run python -m trainer.account set-password brendan
"""

import argparse
import getpass
import sys
from pathlib import Path

from . import auth
from .db import DEFAULT_DB, connect


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show users, trial counts, and whether they have a password")
    sp = sub.add_parser("set-password", help="give a user a password so they can sign in")
    sp.add_argument("name")
    sp.add_argument("--rename-to", help="also give the row a login-friendly username")
    args = ap.parse_args(argv)

    conn = connect(args.db)

    if args.cmd == "list":
        for row in conn.execute(
            "SELECT name, attempts, password_hash IS NOT NULL AS claimed FROM users ORDER BY id"
        ):
            kind = "account" if row["claimed"] else "guest"
            print(f"{row['name']:<40} {row['attempts']:>6} trials  {kind}")
        return 0

    user = auth.find_by_username(conn, args.name)
    if user is None:
        print(f"no user named {args.name!r}", file=sys.stderr)
        return 1
    try:
        name = auth.check_username(args.rename_to or user["name"])
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
    print(f"{name} can now sign in ({user['attempts']} trials preserved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
