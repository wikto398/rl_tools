# syntax=docker/dockerfile:1.7
#
# Reusable, engine-agnostic base image for RL projects built on `rl_tools`.
#
# Build context is the framework submodule (it needs pyproject.toml/uv.lock):
#
#     docker build -f rl_tools/docker/Dockerfile -t rl-base:rocm7.2.4-py3.12 rl_tools/
#
# It contains no engine and no game code: the ROCm/PyTorch userspace, a Python
# environment with this project's dependencies in /opt/venv, and build tooling.
# Engine layers extend it via `FROM ${BASE_IMAGE}` — see
# rl_tools/docker/godot/Dockerfile (and, later, unity/ etc.), or run
# rl_tools/docker/build.sh to build base + godot in order.
#
# ROCm matches a typical host userspace; the torch wheels themselves are pinned
# by uv.lock to the +rocm7.1 local version declared in pyproject.toml.

ARG ROCM_VERSION=7.2.4
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.10.0

FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py${PYTHON_VERSION}_pytorch_release_${TORCH_VERSION}

LABEL org.opencontainers.image.title="rl-base" \
      org.opencontainers.image.description="Engine-agnostic RL base: ROCm + PyTorch + uv + rl_tools dependencies"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# --- build tooling -----------------------------------------------------------
# Engine-agnostic on purpose: engine layers add their own runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*

# --- Python environment ------------------------------------------------------
# Dependencies only (the framework/game source is provided at runtime). The uv
# cache mount keeps repeat builds fast without baking the cache into the image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /opt/build
COPY pyproject.toml uv.lock .python-version /opt/build/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project

WORKDIR /app
CMD ["python", "--version"]
