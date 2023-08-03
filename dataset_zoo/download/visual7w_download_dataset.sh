#!/usr/bin/env bash

V7W_DB_NAME=v7w_telling
DOWNLOAD_DIR=$HOME/scratch/decompose-composition/datasets/visual7w

V7W_URL="http://ai.stanford.edu/~yukez/papers/resources/dataset_${V7W_DB_NAME}.zip"
V7W_PATH="dataset_${V7W_DB_NAME}.json"

# create download directory if it does not exist
if [ ! -d "$DOWNLOAD_DIR" ]; then
  echo "Creating download directory: $DOWNLOAD_DIR"
  mkdir -p "$DOWNLOAD_DIR"
fi

if [ -f "${DOWNLOAD_DIR}/dataset.json" ]; then
  echo "Dataset already exists."
  read -p "Would you like to update the dataset file? [y/n] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    echo "Bye!"
    exit
  fi
fi

echo "Downloading ${V7W_DB_NAME} dataset..."
wget -q $V7W_URL -O "${DOWNLOAD_DIR}/dataset.zip"
unzip -j "${DOWNLOAD_DIR}/dataset.zip" -d $DOWNLOAD_DIR
rm "${DOWNLOAD_DIR}/dataset.zip"
mv "${DOWNLOAD_DIR}/${V7W_PATH}" "${DOWNLOAD_DIR}/dataset.json"
echo "Done."


V7W_IMG_URL=http://vision.stanford.edu/yukezhu/visual7w_images.zip
DOWNLOAD_DIR=$HOME/scratch/decompose-composition/datasets/visual7w

echo "Downloading Visual7W images..."
wget -P $DOWNLOAD_DIR $V7W_IMG_URL
unzip -d $DOWNLOAD_DIR $DOWNLOAD_DIR/visual7w_images.zip
rm $DOWNLOAD_DIR/visual7w_images.zip
echo "Done."
