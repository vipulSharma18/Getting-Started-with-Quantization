FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=${HF_HOME}
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# add keyring for gh/github cli for git creds management
RUN (type -p wget >/dev/null || (apt-get update && apt-get install wget -y)) \
&& mkdir -p -m 755 /etc/apt/keyrings \
&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& mkdir -p -m 755 /etc/apt/sources.list.d \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    vim \
    git \
    gh \
    python3 \
    python3-pip \
    ca-certificates \
    openssh-client \
    openssh-server \
    procps \
    net-tools \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml
ADD .python-version /app/.python-version

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

RUN mkdir -p ${TRANSFORMERS_CACHE}

ENTRYPOINT [ "/app/entrypoint.sh" ]

CMD ["/bin/bash", "-c", "echo 'Container Started' && sleep infinity"]