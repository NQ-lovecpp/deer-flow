# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ARG APT_MIRROR=
ARG COMFYUI_GIT_REF=0c63b4f6
ARG COMFYUI_REPO_URL=https://github.com/comfyanonymous/ComfyUI.git
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

RUN set -eux; \
    if [ -n "${APT_MIRROR}" ]; then \
      find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -print0 \
        | xargs -0 sed -i \
          -e "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
          -e "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
          -e "s|http://ports.ubuntu.com/ubuntu-ports|${APT_MIRROR}|g" \
          -e "s|https://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
          -e "s|https://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
          -e "s|https://ports.ubuntu.com/ubuntu-ports|${APT_MIRROR}|g"; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      python3-pip; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
      git -c http.version=HTTP/1.1 clone --filter=blob:none "${COMFYUI_REPO_URL}" ComfyUI && break; \
      status="$?"; \
      rm -rf ComfyUI; \
      if [ "$attempt" = 5 ]; then exit "$status"; fi; \
      sleep $((attempt * 5)); \
    done; \
    cd ComfyUI; \
    if git cat-file -e "${COMFYUI_GIT_REF}^{commit}" 2>/dev/null; then \
      git checkout "${COMFYUI_GIT_REF}"; \
    else \
      git -c http.version=HTTP/1.1 fetch --depth 1 origin "${COMFYUI_GIT_REF}"; \
      git checkout FETCH_HEAD; \
    fi

WORKDIR /app/ComfyUI

RUN set -eux; \
    python3 -m pip install --upgrade pip setuptools wheel; \
    pip_args=""; \
    if [ -n "${PIP_INDEX_URL}" ]; then pip_args="$pip_args --index-url ${PIP_INDEX_URL}"; fi; \
    if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then pip_args="$pip_args --extra-index-url ${PIP_EXTRA_INDEX_URL}"; fi; \
    if [ -n "${PIP_TRUSTED_HOST}" ]; then pip_args="$pip_args --trusted-host ${PIP_TRUSTED_HOST}"; fi; \
    python3 -m pip install $pip_args -r requirements.txt; \
    mkdir -p models input output user custom_nodes temp; \
    chown -R 1000:1000 /app/ComfyUI

COPY entrypoint.sh /usr/local/bin/comfyui-entrypoint
RUN chmod +x /usr/local/bin/comfyui-entrypoint

EXPOSE 8188

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=6 \
  CMD python3 -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('COMFYUI_PORT', '8188') + '/system_stats', timeout=3).read()"

ENTRYPOINT ["/usr/local/bin/comfyui-entrypoint"]
