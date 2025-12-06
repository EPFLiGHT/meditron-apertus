#!/bin/bash

if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found. Please create one based on .env.example"
    exit 1
fi

# 1. Check if a script was provided
if [ -z "$1" ]; then
    echo "Usage: ./launch.sh <your_sbatch_script.sh>"
    exit 1
fi

SCRIPT_PATH="$1"

# 2. Extract the Job Name defined in the SBATCH script
# We grab the line starting with #SBATCH --job-name and take the last word
JOB_NAME=$(grep -m 1 "#SBATCH --job-name" "$SCRIPT_PATH" | awk '{print $3}')

if [ -z "$JOB_NAME" ]; then
    echo "Error: Could not find #SBATCH --job-name in $SCRIPT_PATH"
    exit 1
fi

# 3. Submit the job and capture the output (e.g., "Submitted batch job 1196537")
SUBMISSION_OUTPUT=$(sbatch "$SCRIPT_PATH")
JOB_ID=$(echo "$SUBMISSION_OUTPUT" | awk '{print $4}')

echo "🚀 Submitted Job: $JOB_ID"
echo "📄 Job Name:      $JOB_NAME"

# 4. Construct the log file path based on your pattern: /reports/R-%x.%j.err
# Note: I am using the path found in your script provided above.
LOG_FILE="$PROJECT_ROOT/reports/R-${JOB_NAME}.${JOB_ID}.err"

echo "Please wait, looking for log file: $LOG_FILE"

# 5. Wait loop: SLURM takes a few seconds to create the file. We wait until it exists.
while [ ! -f "$LOG_FILE" ]; do
    sleep 1
done

echo "✅ Log file found! Tailing now (Press Ctrl+C to stop watching, job will keep running)..."
echo "---------------------------------------------------------------------------------------"

# 6. Tail the file
tail -f "$LOG_FILE"