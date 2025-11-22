#!/bin/bash

# Wait for process 2154381 to finish
PID=2154381

echo "Waiting for process ${PID} to finish..."
while kill -0 ${PID} 2>/dev/null; do
    sleep 3600  # Check every hour
done

echo "Process ${PID} has finished. Starting next training..."
cd "$(dirname "$0")/.." || exit 1
nohup ./shell/train_sp.sh random1 1 > log/train_sp_random1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "New training process started with PID: $!"
