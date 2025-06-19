#!/bin/bash

PROJECT_DIR=$(dirname "$(realpath "$0")")
echo "Project directory: $PROJECT_DIR"

LOG_FILE="$PROJECT_DIR/logs/app.log"
echo "Log file path: $LOG_FILE"

DOWNLOAD_DIR="$PROJECT_DIR/data/testing_video/"
echo "Download directory: $DOWNLOAD_DIR"

if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "Directory $DOWNLOAD_DIR does not exist. Creating it..."
    mkdir -p "$DOWNLOAD_DIR"
    echo "Made directory $DOWNLOAD_DIR"
fi

echo "downloading files to $DOWNLOAD_DIR"

kaggle datasets download saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv
unzip "$PROJECT_DIR/-dfl-bundesliga-460-mp4-videos-in-30sec-csv.zip" -d "$DOWNLOAD_DIR"
rm "$PROJECT_DIR/-dfl-bundesliga-460-mp4-videos-in-30sec-csv.zip"

gdown -O "$DOWNLOAD_DIR/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
gdown -O "$DOWNLOAD_DIR/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"


echo "Done!"
