ARG PIXI_VER=0.59.0
ARG LINUX_VER=24.04
ARG RELEASE_VERSION=v0.2.0
ARG REVISION=latest

FROM ghcr.io/prefix-dev/pixi:${PIXI_VER} AS build

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
  git build-essential ca-certificates tzdata \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir -p /app/.config

ENV NUMBA_CACHE_DIR=/app/.config/numba
ENV MPLCONFIGDIR=/app/.config/matplotlib

RUN pixi install
RUN pixi shell-hook -s bash > /shell-hook

FROM ubuntu:${LINUX_VER} AS neuralift-c360-prep

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
  ca-certificates tzdata \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /shell-hook /app/shell-hook
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
COPY --from=build /app/.config /app/.config

COPY --from=build /app/src /app/src
COPY --from=build /app/configs /app/configs

ARG RELEASE_VERSION
ENV RELEASE_VERSION=${RELEASE_VERSION}

ARG REVISION
ENV GIT_SHA=${REVISION}

ENV NUMBA_CACHE_DIR=/app/.config/numba
ENV MPLCONFIGDIR=/app/.config/matplotlib
ENV NEURALIFT_C360_PREP_IS_CONTAINER=true

ENTRYPOINT ["/app/entrypoint.sh"]
