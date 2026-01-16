#!/bin/bash

# Target directory
TARGET_DIR=~/tests/govdocs
mkdir -p "$TARGET_DIR"

echo "Downloading GovDocs1 zip files to $TARGET_DIR..."

for i in {0..19}; do
    # Format number with leading zeros (000, 001, ..., 019)
    NUM=$(printf "%03d" "$i")
    ZIP_FILE="${NUM}.zip"
    URL="https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/${ZIP_FILE}"
    DEST_DIR="${TARGET_DIR}/${NUM}"
    
    echo "Processing $ZIP_FILE..."
    
    # Download
    wget -q --show-progress "$URL" -O "$ZIP_FILE"
    
    if [ $? -eq 0 ]; then
        # Create destination directory
        mkdir -p "$DEST_DIR"
        
        # Unzip directly into the destination directory
        unzip -q -j "$ZIP_FILE" -d "$DEST_DIR"
        
        # Clean up zip
        rm "$ZIP_FILE"
        echo "Successfully downloaded and extracted $ZIP_FILE to $DEST_DIR"
    else
        echo "Failed to download $ZIP_FILE"
        rm -f "$ZIP_FILE"
    fi
done

echo "Done!"
