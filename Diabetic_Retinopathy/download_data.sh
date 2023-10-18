#!/bin/bash

# Check if the URL argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <URL>"
  exit 1
fi

# Get the URL argument
url="$1"

# Extract the file name from the URL
output_file="archive.zip"

# Download the file using wget
wget -O "$output_file" "$url"
