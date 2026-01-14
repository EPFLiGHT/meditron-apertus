#!/bin/bash

# If argn != 1, exit with error
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <build|export|run>"
    exit 1
fi

PATH_TO_SCRIPT=$(dirname "$0")
PATH_TO_DOCKERFILE="$PATH_TO_SCRIPT/Dockerfile"
ACTION=$1

IMAGE_NAME="meditron-apertus"
EXPORT_PATH="/capstor/store/cscs/swissai/a127/meditron/docker/new_axolotl.sqsh"

if [ "$ACTION" == "build" ]; then
    podman build -t $IMAGE_NAME -f "$PATH_TO_DOCKERFILE" ../
elif [ "$ACTION" == "export" ]; then
    rm -f "$EXPORT_PATH"
    enroot import -o "$EXPORT_PATH" podman://localhost/$IMAGE_NAME:latest
    setfacl -b "$EXPORT_PATH"
    chmod +r "$EXPORT_PATH"
elif [ "$ACTION" == "run" ]; then
    podman run --interactive --tty $IMAGE_NAME /bin/bash
else
    echo "Invalid action: $ACTION. Use build, export, or run."
    exit 1
fi
