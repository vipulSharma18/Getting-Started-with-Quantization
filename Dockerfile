FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/mnt/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    vim \
    git \
    python3 \
    python3-pip \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml
ADD .python-version /app/.python-version

# split the dependency installation from the workspace members' installation
RUN --mount=type=cache,target=/mnt/.cache/uv \
    uv sync --locked --no-install-project

RUN --mount=type=cache,target=/mnt/.cache/uv \
    uv sync --locked --no-install-project --group gemlite

RUN --mount=type=cache,target=/mnt/.cache/uv \
    uv sync --locked --no-install-project --group tensorRT

RUN --mount=type=cache,target=/mnt/.cache/uv \
    uv sync --locked --no-install-project --group torchao

ADD . /app

RUN --mount=type=cache,target=/mnt/.cache/uv \
    uv sync --locked

CMD ["bash"]