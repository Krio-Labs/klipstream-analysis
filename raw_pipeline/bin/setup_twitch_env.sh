#!/bin/bash
# Script to set up environment variables for TwitchDownloaderCLI_mac

# Create the directory for .NET bundle extraction
mkdir -p ~/.dotnet/bundle_extract

# Export the environment variable
export DOTNET_BUNDLE_EXTRACT_BASE_DIR=~/.dotnet/bundle_extract

# Echo a confirmation message
echo "DOTNET_BUNDLE_EXTRACT_BASE_DIR set to $DOTNET_BUNDLE_EXTRACT_BASE_DIR"

# If a command was provided, run it with the environment variable set
if [ $# -gt 0 ]; then
    echo "Running command: $@"
    "$@"
fi
