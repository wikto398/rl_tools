#!/usr/bin/env bash
#
# Build the reusable framework images, in order: base -> godot.
#
#     rl_tools/docker/build.sh
#     BASE_IMAGE=rl-base:rocm7.2.4-py3.12 GODOT_IMAGE=rl-godot:4.7.2 \
#         rl_tools/docker/build.sh
#
# A game image then builds ON TOP of ${GODOT_IMAGE} (see the game repo's
# docker/Dockerfile and docker/compose.yml).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(dirname "$here")" # rl_tools/

BASE_IMAGE="${BASE_IMAGE:-rl-base:rocm7.2.4-py3.12}"
GODOT_IMAGE="${GODOT_IMAGE:-rl-godot:4.7.2}"
GODOT_VERSION="${GODOT_VERSION:-4.7.2}"

echo "==> base:  ${BASE_IMAGE}"
docker build -f "${here}/Dockerfile" -t "${BASE_IMAGE}" "${framework_root}"

echo "==> godot: ${GODOT_IMAGE} (FROM ${BASE_IMAGE})"
docker build -f "${here}/godot/Dockerfile" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "GODOT_VERSION=${GODOT_VERSION}" \
    -t "${GODOT_IMAGE}" "${here}/godot"

echo "==> done: ${BASE_IMAGE}, ${GODOT_IMAGE}"
