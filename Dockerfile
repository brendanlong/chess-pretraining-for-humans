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

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 TRAINER_DB=/data/items.db

# Everything that can reach this port is either the Fly proxy or inside our own
# private network, so the forwarded client address can be believed — which is
# what makes the signup limiter count real addresses instead of counting the
# whole site as one, and lets the session cookie pick `Secure` up from the
# scheme the browser actually used.
ENV FORWARDED_ALLOW_IPS="*"

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "trainer.server:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
