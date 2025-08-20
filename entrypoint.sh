#!/bin/bash
set -e

# Ensure libcuda.so exists
if [ ! -f /lib/x86_64-linux-gnu/libcuda.so ] && [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    echo "Creating libcuda.so symlink..."
    ln -s /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so
    ldconfig
fi

# Pass control to CMD
exec "$@"
