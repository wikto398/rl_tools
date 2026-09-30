# syntax=docker/dockerfile:1.7
#
# Godot engine layer: extends the engine-agnostic base with the Godot runtime.
#
#     docker build -f rl_tools/docker/godot/Dockerfile \
#         --build-arg BASE_IMAGE=rl-base:rocm7.2.4-py3.12 \
#         -t rl-godot:4.7.2 rl_tools/docker/godot
#
# (rl_tools/docker/build.sh builds this together with the base.) A game image
# then extends this one: `FROM rl-godot:<version>`. Other engines follow the
# same pattern in a sibling directory (e.g. rl_tools/docker/unity/Dockerfile).

ARG BASE_IMAGE=rl-base:rocm7.2.4-py3.12
FROM ${BASE_IMAGE}

ARG GODOT_VERSION=4.7.2

LABEL org.opencontainers.image.title="rl-godot" \
      org.opencontainers.image.description="Godot runtime on top of rl-base" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

# Runtime libraries the Godot binary links even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfontconfig1 libfreetype6 libgl1 libglu1-mesa \
        libxcursor1 libxinerama1 libxrandr2 libxi6 libx11-6 libxext6 \
        libxrender1 libasound2t64 libpulse0 \
    && rm -rf /var/lib/apt/lists/*

# The standard linux build supports `--headless`; no separate headless download.
RUN curl -fsSL -o /tmp/godot.zip \
        "https://github.com/godotengine/godot/releases/download/${GODOT_VERSION}-stable/Godot_v${GODOT_VERSION}-stable_linux.x86_64.zip" \
    && unzip -q /tmp/godot.zip -d /tmp/godot \
    && install -m 0755 "/tmp/godot/Godot_v${GODOT_VERSION}-stable_linux.x86_64" /usr/local/bin/godot \
    && rm -rf /tmp/godot /tmp/godot.zip \
    && godot --version

CMD ["godot", "--version"]
