#!/usr/bin/env bash

# Usage:
#   ./clean_subfolders.sh /path/to/dir keep1 keep2 keep3 ...

set -e

TARGET_DIR="/homes/gcasari/bigbrain/work_data/logs"
shift
KEEP_LIST=("$@")

# Convert keep list into a pattern for easy matching
# Example: keep1 keep2 → ^(keep1|keep2)$
KEEP_REGEX="^($(printf "%s|" "${KEEP_LIST[@]}" | sed 's/|$//'))$"

# Loop over all subdirectories
for d in "$TARGET_DIR"/*/ ; do
    d=$(basename "$d")

    if [[ ! "$d" =~ $KEEP_REGEX ]]; then
        echo "Removing: $TARGET_DIR/$d"
        rm -rf "$TARGET_DIR/$d"
    else
        echo "Keeping:  $TARGET_DIR/$d"
    fi
done