FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH="/root/.local/bin/:$PATH"

EXPOSE 22/tcp 80/tcp 443/tcp 8080/tcp 
EXPOSE 22/udp 80/udp 443/udp 8080/udp

ADD https://astral.sh/uv/install.sh /uv-installer.sh

# add keyring for gh/github cli for git creds management
RUN (type -p wget >/dev/null || (apt-get update && apt-get install wget -y)) \
&& mkdir -p -m 755 /etc/apt/keyrings \
&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& mkdir -p -m 755 /etc/apt/sources.list.d \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& apt-get update \ 
&& apt-get install -y --no-install-recommends \
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
    graphviz \
&& rm -rf /var/lib/apt/lists/* \
&& sh /uv-installer.sh \
&& rm /uv-installer.sh \
&& mkdir -p ${HF_HOME}

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked 

ENTRYPOINT [ "/app/entrypoint.sh" ]

CMD ["/bin/bash", "-c", "echo '[dockerfile] Container Started' && sleep infinity"]