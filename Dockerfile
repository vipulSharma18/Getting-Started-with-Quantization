FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy

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
# split the dependency installation from the workspace members' installation
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group gemlite

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group tensorRT

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group torchao

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD ["bash"]