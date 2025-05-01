#!/bin/bash

base_url="https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000070_BigBrain-selected_1um_scans_pub/v1.0/aligned/"
save_path="/work/bolelli_synthetic/BigBrain/high_res_aligned/"

# Get all .json filenames and extract sorted IDs
affine_ids=$(ls "$save_path"/*.json 2>/dev/null | sed -E 's/.*B20_([0-9]+)_affine\.json/\1/' | sort)

for id in $affine_ids; do
    filename="B20_${id}.tif"
    url="${base_url}${filename}"
    outpath="${save_path}${filename}"

    if [ -f "$outpath" ]; then
        echo "Already downloaded: $filename"
        continue
    fi

    echo "DOWNLOADING: $filename"

    if wget "$url" -O "$outpath"; then
        echo "Downloaded: $filename"
    else
        echo "Failed to download: $filename"
    fi
done