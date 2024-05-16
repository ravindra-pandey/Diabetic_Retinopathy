if [ $# -ne 1 ]; then
  echo "Usage: $0 <URL>"
  exit 1
fi

url="$1"

output_file="archive.zip"

curl -o "$output_file" "$url"
