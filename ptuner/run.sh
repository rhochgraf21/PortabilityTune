#!/bin/bash
image_name="python-ptuner"

# system-independent way to get absolute path
get_abs_filename() {
  # $1 : relative filename
  filename=$1
  parentdir=$(dirname "${filename}")

  if [ -d "${filename}" ]; then
      echo "$(cd "${filename}" && pwd)"
  elif [ -d "${parentdir}" ]; then
    echo "$(cd "${parentdir}" && pwd)/$(basename "${filename}")"
  fi
}

# check if two arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <path/to/config/file> <path/to/output/file>"
  exit 1
fi

# get the absolute paths to the files
config=$(get_abs_filename "$1")
outfile=$(get_abs_filename "$2")

# read the database path from config.yml 
value=$(grep "database_path:" config.yml | awk '{print $2}')
db=$(get_abs_filename "$value")

# run the docker image and mount the files as volumes
docker run --mount type=bind,source="$config",target=/mount/"$1" \
           --mount type=bind,source="$outfile",target=/mount/"$2" \
           --mount type=bind,source="$db",target=/mount/$value \
           $image_name /mount/"$1" /mount/"$2"