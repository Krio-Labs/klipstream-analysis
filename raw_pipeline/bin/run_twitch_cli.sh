#!/bin/bash
# Wrapper script for TwitchDownloaderCLI_mac that sets the required environment variables

# Set up .NET bundle extract directory for TwitchDownloaderCLI_mac
DOTNET_DIR=~/.dotnet/bundle_extract

# Create the directory if it doesn't exist
if [ ! -d "$DOTNET_DIR" ]; then
    echo "Creating .NET bundle extract directory at $DOTNET_DIR"
    mkdir -p "$DOTNET_DIR"
fi

# Set the environment variable
export DOTNET_BUNDLE_EXTRACT_BASE_DIR="$DOTNET_DIR"
echo "Set DOTNET_BUNDLE_EXTRACT_BASE_DIR to $DOTNET_BUNDLE_EXTRACT_BASE_DIR"

# Run TwitchDownloaderCLI_mac with all arguments passed to this script
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$SCRIPT_DIR/TwitchDownloaderCLI_mac" "$@"
