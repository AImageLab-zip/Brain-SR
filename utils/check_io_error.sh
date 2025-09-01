#!/bin/bash
# check_io.sh DIR

DIR="$1"
total=0
bad=0

for f in "$DIR"/*; do
    [ -f "$f" ] || continue
    ((total++))
    if ! head -c 1 "$f" >/dev/null 2>&1; then
        echo "Unreadable: $f"
        ((bad++))
    fi
done

echo "----"
echo "Total files:     $total"
echo "Unreadable:      $bad"
if [ "$total" -gt 0 ]; then
    pct=$(awk -v b="$bad" -v t="$total" 'BEGIN{ printf "%.2f", (b*100)/t }')
    echo "Percentage bad:  $pct %"
fi