#!/bin/bash

# Define the URL and the destination directory
URL="https://archive.ics.uci.edu/static/public/52/ionosphere.zip"
DEST_DIR="data"
ZIP_FILE="ionosphere.zip"

# Step 1: Download the zip file using curl
echo "Downloading ionosphere.zip from $URL..."
curl -o $ZIP_FILE $URL

# Step 2: Create the 'data' directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating directory: $DEST_DIR"
    mkdir $DEST_DIR
fi

# Step 3: Unzip the downloaded zip file
echo "Extracting $ZIP_FILE..."
unzip $ZIP_FILE

# Step 4: Move the 'ionosphere.data' file into the 'data' directory
if [ -f "ionosphere.data" ]; then
    echo "Moving ionosphere.data to $DEST_DIR"
    mv ionosphere.data $DEST_DIR/
else
    echo "Error: ionosphere.data not found in the zip file."
    exit 1
fi

# Step 5: Clean up by removing the zip file
echo "Cleaning up: Removing $ZIP_FILE"
rm $ZIP_FILE

echo "Download and extraction completed successfully."
rm Index ionosphere.names