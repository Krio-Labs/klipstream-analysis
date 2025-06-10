#!/usr/bin/env python3
"""
Check Deployment Environment Variables

This script checks if all required environment variables are set for deployment.
"""

import os
from pathlib import Path

def check_deployment_environment():
    """Check if all required environment variables are set"""
    
    print("🔍 Checking Deployment Environment Variables...")
    print("=" * 60)
    
    # Required environment variables
    required_vars = {
        "CONVEX_URL": "Convex database URL",
        "DEEPGRAM_API_KEY": "Deepgram API key (for fallback transcription)",
        "NEBIUS_API_KEY": "Nebius API key (for sentiment analysis)"
    }

    # Optional but recommended
    optional_vars = {
        "CONVEX_DEPLOY_KEY": "Convex deployment key (not needed for client connections)",
        "GOOGLE_APPLICATION_CREDENTIALS": "Google Cloud service account key file",
        "GCS_PROJECT_ID": "Google Cloud Storage project ID"
    }
    
    missing_required = []
    missing_optional = []
    
    print("📋 Required Environment Variables:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "TOKEN" in var:
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                print(f"   ✅ {var}: {masked_value}")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: NOT SET ({description})")
            missing_required.append(var)
    
    print(f"\n📋 Optional Environment Variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ⚠️  {var}: NOT SET ({description})")
            missing_optional.append(var)
    
    # Check for .env file
    print(f"\n📄 Environment File Check:")
    env_file = Path(".env")
    if env_file.exists():
        print(f"   ✅ .env file found")
        print(f"   📝 Loading environment variables from .env...")
        
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print(f"   ✅ .env file loaded successfully")
            
            # Re-check after loading .env
            print(f"\n🔄 Re-checking after loading .env:")
            missing_after_env = []
            for var in missing_required:
                value = os.getenv(var)
                if value:
                    print(f"   ✅ {var}: Now set from .env")
                else:
                    missing_after_env.append(var)
            
            missing_required = missing_after_env
            
        except ImportError:
            print(f"   ⚠️  python-dotenv not installed, cannot load .env file")
        except Exception as e:
            print(f"   ❌ Error loading .env file: {e}")
    else:
        print(f"   ⚠️  .env file not found")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 Environment Check Summary:")
    
    if not missing_required:
        print(f"✅ All required environment variables are set!")
        print(f"🚀 Ready for deployment!")
        
        if missing_optional:
            print(f"\n⚠️  Optional variables missing: {len(missing_optional)}")
            for var in missing_optional:
                print(f"   • {var}")
            print(f"   These are optional but may improve functionality")
        
        return True
    else:
        print(f"❌ Missing required environment variables: {len(missing_required)}")
        for var in missing_required:
            print(f"   • {var}")
        
        print(f"\n🔧 How to set environment variables:")
        print(f"   1. Create a .env file in the project root:")
        print(f"      touch .env")
        print(f"   2. Add the missing variables:")
        for var in missing_required:
            print(f"      echo '{var}=your_value_here' >> .env")
        print(f"   3. Or export them in your shell:")
        for var in missing_required:
            print(f"      export {var}=your_value_here")
        
        return False

def check_gcloud_auth():
    """Check if gcloud is authenticated"""
    print(f"\n🔐 Checking Google Cloud Authentication:")
    
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            accounts = result.stdout.strip().split('\n')
            print(f"   ✅ Authenticated accounts: {len(accounts)}")
            for account in accounts:
                print(f"      • {account}")
            return True
        else:
            print(f"   ❌ No active gcloud authentication found")
            print(f"   🔧 Run: gcloud auth login")
            return False
            
    except FileNotFoundError:
        print(f"   ❌ gcloud CLI not found")
        print(f"   🔧 Install: https://cloud.google.com/sdk/docs/install")
        return False
    except Exception as e:
        print(f"   ❌ Error checking gcloud auth: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Pre-Deployment Environment Check")
    print("=" * 60)
    
    env_ready = check_deployment_environment()
    gcloud_ready = check_gcloud_auth()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Deployment Readiness:")
    
    if env_ready and gcloud_ready:
        print(f"✅ READY FOR DEPLOYMENT!")
        print(f"🚀 You can now run: ./deploy_cloud_run_gpu.sh")
        return 0
    else:
        print(f"❌ NOT READY FOR DEPLOYMENT")
        if not env_ready:
            print(f"   • Fix missing environment variables")
        if not gcloud_ready:
            print(f"   • Set up Google Cloud authentication")
        print(f"   • Re-run this script after fixing issues")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
