#!/bin/bash
# Simple script to kill all sp_generalize related processes

echo "========================================"
echo "Killing sp_generalize processes"
echo "========================================"

# Find and display processes first
echo "Finding processes with 'sp_generalize@'..."
PIDS=$(ps aux | grep "sp_generalize@" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No sp_generalize processes found."
    exit 0
fi

# Count and display
COUNT=$(echo "$PIDS" | wc -w)
echo "Found $COUNT processes to kill:"
ps aux | grep "sp_generalize@" | grep -v grep | awk '{printf "  PID: %-7s CMD: %s\n", $2, $11}'

echo ""
echo "Killing processes..."

# Kill all found processes
for pid in $PIDS; do
    kill -9 $pid 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ Killed PID: $pid"
    else
        echo "  ✗ Failed to kill PID: $pid"
    fi
done

sleep 1

# Check if any remain
echo ""
echo "Checking for remaining processes..."
REMAINING=$(ps aux | grep "sp_generalize@" | grep -v grep)

if [ -z "$REMAINING" ]; then
    echo "✓ All sp_generalize processes killed successfully!"
else
    echo "⚠ Some processes still remain:"
    echo "$REMAINING" | awk '{printf "  PID: %-7s CMD: %s\n", $2, $11}'
    echo ""
    echo "You can try running this script again or manually kill with:"
    echo "  kill -9 <PID>"
fi

echo "========================================"

