"""Talking to lichess.org: the OAuth sign-in, and reading linked ratings.

Lichess OAuth is PKCE with a public client, so there is no secret to hold and
no app to register: `client_id` is any stable name, and the redirect URI only
has to be the same in both halves of one flow.

The grant is spent on one question — who is this? — and revoked straight
away. Nothing kept afterwards needs it: a linked account's ratings are public,
read by id from `/api/users`, so holding the token would only mean a database
read hands out a Lichess credential.

Every call here is outbound network on a request thread or the refresher's, so
each one has a short timeout and every failure surfaces as `LichessError`.
"""

import base64
import hashlib
import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request

HOST = os.environ.get("LICHESS_HOST", "https://lichess.org")
CLIENT_ID = os.environ.get("LICHESS_CLIENT_ID", "chess-pretraining")
TIMEOUT_S = 10
# What `/api/users` takes in one request.
BATCH = 300


class LichessError(Exception):
    """Lichess refused, or couldn't be reached."""


def new_flow() -> tuple[str, str]:
    """(state, code_verifier), both fresh and unguessable."""
    return secrets.token_urlsafe(24), secrets.token_urlsafe(48)


def challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def authorize_url(redirect_uri: str, state: str, verifier: str) -> str:
    # No scope: reading the account a token belongs to needs none, and asking
    # for nothing is what makes the consent screen say we can do nothing.
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge(verifier),
            "state": state,
        }
    )
    return f"{HOST}/oauth?{query}"


def _call(
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    token: str | None = None,
):
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{HOST}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as res:
            raw = res.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LichessError(f"{method} {path}: {e}") from e
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        raise LichessError(f"{method} {path}: not JSON") from e


def exchange(code: str, verifier: str, redirect_uri: str) -> str:
    """Trade an authorization code for an access token."""
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
        }
    ).encode()
    reply = _call("POST", "/api/token", body=body, content_type="application/x-www-form-urlencoded")
    token = reply.get("access_token") if isinstance(reply, dict) else None
    if not token:
        raise LichessError("token endpoint returned no access_token")
    return token


def account(token: str) -> dict:
    """The account a token belongs to, as `/api/account` describes it."""
    reply = _call("GET", "/api/account", token=token)
    if not isinstance(reply, dict) or not reply.get("id") or not reply.get("username"):
        raise LichessError("account endpoint returned no user")
    return reply


def revoke(token: str) -> None:
    _call("DELETE", "/api/token", token=token)


def users(ids: list[str]) -> list[dict]:
    """Public records for up to `BATCH` ids. Closed accounts come back marked
    `disabled`, and unknown ones not at all.

    `profile=false` because the profile is free text — bio, real name,
    location — and none of it is a rating."""
    reply = _call(
        "POST",
        "/api/users?profile=false",
        body=",".join(ids).encode(),
        content_type="text/plain",
    )
    if not isinstance(reply, list):
        raise LichessError("users endpoint returned no list")
    return [u for u in reply if isinstance(u, dict) and u.get("id")]


def perfs(user: dict) -> str:
    """The ratings to keep from a user record, as stored: every perf whole.

    Whole for the reason `items.gap_ladder` is kept whole — which perf, and
    which of its fields, is the transfer measure is a question for analysis,
    and only what was recorded at the time can answer it later.
    """
    return json.dumps(user.get("perfs") or {}, sort_keys=True)
