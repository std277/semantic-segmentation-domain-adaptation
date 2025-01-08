#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Check if Train, Val, and Test directories exist
if [ ! -d "data/Train" ] || [ ! -d "data/Val" ] || [ ! -d "data/Test" ]; then
  # Download the zip files
  wget https://zenodo.org/records/5706578/files/Train.zip
  wget https://zenodo.org/records/5706578/files/Val.zip
  wget https://zenodo.org/records/5706578/files/Test.zip

  # Unzip the files into the data directory
  unzip Train.zip -d data
  unzip Val.zip -d data
  unzip Test.zip -d data

  # Remove the zip files
  rm Train.zip
  rm Val.zip
  rm Test.zip
else
  echo "Directories Train, Val, and Test already exist."
fi