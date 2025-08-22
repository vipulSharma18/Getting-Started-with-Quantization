#!/bin/bash

echo "[entrypoint] running sshd checks"

echo "[entrypoint] ensuring /var/run/sshd and /etc/ssh/ exists"
mkdir -p /var/run/sshd
mkdir -p /etc/ssh/

echo "[entrypoint] ensuring host keys exist in /etc/ssh:"
ls -l /etc/ssh/ || echo "[entrypoint] /etc/ssh/ does not exist, i.e., no host keys found"

echo "[entrypoint] generating host keys if any are missing..."
ssh-keygen -A || echo "[entrypoint] ERROR: ssh-keygen failed"

echo "[entrypoint] new keys in /etc/ssh:"
ls -l /etc/ssh/ || echo "[entrypoint] /etc/ssh/ does not exist, i.e., no host keys found"

echo "[entrypoint] Starting ssh service after generating keys..."
/usr/sbin/sshd || echo "[entrypoint] ERROR: failed to start ssh service"

echo "[entrypoint] Ensuring existence and listing contents of ~/.ssh/:"
mkdir -p ~/.ssh/
chmod 700 ~/.ssh
echo "[entrypoint] manually adding public ssh keys in the ssh_authkeys file as a failcheck"
cat /app/ssh_authkeys >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
ls -l ~/.ssh/ || echo "[entrypoint] ~/.ssh/ does not exist, i.e., no user auth keys found"

echo "[entrypoint] entrypoint sshd checks complete"

echo "[entrypoint] starting app setup"

# Ensure libcuda.so exists
if [ ! -f /lib/x86_64-linux-gnu/libcuda.so ] && [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    echo "[entrypoint] Creating libcuda.so symlink..."
    ln -s /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so
    ldconfig
fi

echo "[entrypoint] starting HF download."

source /app/.venv/bin/activate
hf download --repo-type model unsloth/Meta-Llama-3.1-8B-Instruct

echo "[entrypoint] hf download complete."

echo "[entrypoint] setting up gemlite env"

uv sync --locked --group gemlite

echo "[entrypoint] gemlite environment setup complete"

echo "[entrypoint] entrypoint script complete"

exec "$@"