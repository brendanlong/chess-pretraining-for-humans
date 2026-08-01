# The server only reads the item bank and writes responses; Stockfish and zstd
# belong to the offline pipeline and are deliberately not installed here.

FROM litestream/litestream:0.5.15 AS litestream

# Bundles and minifies web/ into web-dist/. Node lives only in this stage: the
# runtime serves the output and has no idea it was built.
FROM node:22-slim AS web
WORKDIR /app
COPY package.json package-lock.json ./
COPY scripts/ scripts/
RUN npm ci
COPY web/ web/
# `vendor` again rather than trusting npm's postinstall to have done it: that
# ran before web/ was copied, and which order two COPY layers land in is not
# something this should quietly depend on.
RUN npm run vendor && npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
# only-system: a uv-managed interpreter would be downloaded into this stage and
# the venv would point at a path the runtime stage doesn't have.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-system
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen, not --locked, for the same reason CI uses it: uv.lock records an
# `exclude-newer` from the authoring machine that a build host doesn't have.
RUN uv sync --frozen --no-dev

FROM python:3.14-slim-bookworm
COPY --from=litestream /usr/local/bin/litestream /usr/local/bin/litestream
# The server reads the code and writes one directory. Running it as root means a
# code-execution bug also gets to rewrite the image, the database, and
# Litestream's replication metadata. The entrypoint drops to this uid — it can't
# be a `USER` line, because only root can chown the volume, and Fly mounts that
# owned by root long after the build.
RUN useradd --system --uid 10001 --shell /usr/sbin/nologin trainer
WORKDIR /app
COPY --from=build /app/.venv .venv
COPY trainer/ trainer/
# Only the built tree. With no web/ beside it there is nothing for the server
# to pick the wrong one of, and the sources never reach the image.
COPY --from=web /app/web-dist web-dist/
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
