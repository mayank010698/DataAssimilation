#!/bin/bash
# List dataset directories that are len80 and dim 10 (identifiable by len80 and comp10of10).
# Saves to a txt file. Uses same default output dir as generate.py.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit

OUTPUT_DIR="${1:-/data/da_outputs/datasets}"
# Default: save list in project root
OUTFILE="${2:-$SCRIPT_DIR/../datasets_len80.txt}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Output directory does not exist: $OUTPUT_DIR" >&2
    exit 1
fi

# List subdirs: len80, dimensionality 10 (comp10of10), and have data.h5 (complete datasets)
> "$OUTFILE"
for d in "$OUTPUT_DIR"/*/; do
    name=$(basename "$d")
    if [[ "$name" == *len80* && "$name" == *comp10of10* && -f "$d/data.h5" ]]; then
        echo "$d" >> "$OUTFILE"
    fi
done

count=$(wc -l < "$OUTFILE")
echo "Wrote $count dataset path(s) to $OUTFILE"
cat "$OUTFILE"
