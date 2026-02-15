FROM python:3.10-slim

WORKDIR /app

# Copy only dependency files first → enables strong caching
COPY pyproject.toml uv.lock/

# Install uv (fastest way: copy binary from official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Optional: these env vars make uv faster & cleaner in Docker
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install locked dependencies (no dev, frozen = trust the lockfile)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Now copy the actual application code
COPY . .

CMD ["uv", "run", "gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]