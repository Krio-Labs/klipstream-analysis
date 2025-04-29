#!/usr/bin/env python3
"""
Test script for the Twitch Analysis Cloud Function
"""

import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_local_function():
    """Test the cloud function running locally"""
    url = "http://localhost:8080"

    # Get the Twitch video URL from command line or use a default
    twitch_url = input("Enter Twitch video URL (must already exist in your database): ")
    if not twitch_url:
        twitch_url = "https://www.twitch.tv/videos/2442637981"  # Default video

    # Prepare the request payload
    payload = {
        "url": twitch_url
    }

    print("\nIMPORTANT: The Twitch video must already exist in your database before calling this function.")

    print(f"Sending request to {url} with payload:")
    print(json.dumps(payload, indent=2))

    # Send the request
    try:
        response = requests.post(url, json=payload)

        # Print the response
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

def test_deployed_function():
    """Test the deployed cloud function"""
    # Get the deployed function URL
    function_url = input("Enter deployed function URL: ")

    # Get the Twitch video URL from command line or use a default
    twitch_url = input("Enter Twitch video URL (must already exist in your database): ")
    if not twitch_url:
        twitch_url = "https://www.twitch.tv/videos/2442637981"  # Default video

    # Prepare the request payload
    payload = {
        "url": twitch_url
    }

    print("\nIMPORTANT: The Twitch video must already exist in your database before calling this function.")

    print(f"Sending request to {function_url} with payload:")
    print(json.dumps(payload, indent=2))

    # Send the request
    try:
        response = requests.post(function_url, json=payload)

        # Print the response
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Twitch Analysis Cloud Function Test")
    print("==================================")
    print("1. Test local function (http://localhost:8080)")
    print("2. Test deployed function")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        test_local_function()
    elif choice == "2":
        test_deployed_function()
    else:
        print("Invalid choice. Exiting.")
