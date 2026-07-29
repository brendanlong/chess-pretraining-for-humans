# The server only reads the item bank and writes responses; Stockfish and zstd
# belong to the offline pipeline and are deliberately not installed here.

FROM litestream/litestream:0.5.15 AS litestream

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
# only-system: a uv-managed interpreter would be downloaded into this stage and
# the venv would point at a path the runtime stage doesn't have.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-system
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen, not --locked, for the same reason CI uses it: uv.lock records an
# `exclude-newer` from the authoring machine that a build host doesn't have.
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
COPY --from=litestream /usr/local/bin/litestream /usr/local/bin/litestream
WORKDIR /app
COPY --from=build /app/.venv .venv
COPY trainer/ trainer/
COPY web/ web/
COPY deploy/litestream.yml /etc/litestream.yml
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh

# PYTHONPATH because the project isn't an installed package — it's imported
# from the working directory, and `fly ssh console` doesn't promise to land in
# one. Without it the item-bank refresh would fail on the remote half only.
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app PYTHONUNBUFFERED=1 TRAINER_DB=/data/items.db

# This is about the *scheme*, not the address: it lets the session cookie pick
# `Secure` up from the request the browser actually made, and a caller can only
# lie about that for its own cookie. The address uvicorn derives from the same
# headers is not trustworthy (see client_key), which is why the rate limiter
# reads CLIENT_IP_HEADER rather than the socket.
ENV FORWARDED_ALLOW_IPS="*"

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "trainer.server:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
