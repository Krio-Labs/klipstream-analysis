# Testing Documentation

This document describes the test files in the KlipStream Analysis project and how to use them.

## 🧪 Available Test Files

### 1. `test_fastapi_subprocess_fix.py` ⭐ **CRITICAL**

**Purpose**: Tests the FastAPI subprocess wrapper fix for TwitchDownloaderCLI compatibility.

**What it tests**:
- Environment variable inheritance in FastAPI context
- Thread pool execution simulation
- TwitchDownloaderCLI binary compatibility
- Enhanced downloader integration
- Timeout manager integration

**When to run**:
- After making changes to the subprocess wrapper
- Before deploying to Cloud Run
- When debugging "Failure processing application bundle" errors

**Usage**:
```bash
python test_fastapi_subprocess_fix.py
```

**Expected output**: All 3 tests should pass, confirming the FastAPI subprocess fix is working.

---

### 2. `test_cloud_run_api.py` 🌐 **USEFUL**

**Purpose**: Tests the deployed Cloud Run API endpoints.

**What it tests**:
- Health check endpoint
- API documentation accessibility
- Legacy endpoint compatibility (`/api/v1/analyze`)
- New analysis endpoint (`/api/v1/analysis`)
- Enhanced error handling
- Status monitoring capabilities

**When to run**:
- After deploying to Cloud Run
- When validating API functionality
- For production health checks

**Usage**:
```bash
python test_cloud_run_api.py
```

**Expected output**: 80%+ success rate indicates the API is functioning correctly.

---

## 🚀 Test Execution Order

For comprehensive testing, run tests in this order:

1. **Local Environment Setup**:
   ```bash
   python test_fastapi_subprocess_fix.py
   ```

2. **Production API Validation** (after deployment):
   ```bash
   python test_cloud_run_api.py
   ```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
- Ensure you're running from the project root directory
- Activate the virtual environment: `source .venv/bin/activate`

**2. Binary Not Found Errors**
- Check that Git LFS files are downloaded: `git lfs pull`
- Verify binary permissions on macOS: `chmod +x raw_pipeline/bin/TwitchDownloaderCLI_mac`

**3. Environment Variable Issues**
- Source the environment: `source .envrc` (if using direnv)
- Check `.env.yaml` file exists with required variables

**4. Cloud Run API Tests Failing**
- Verify the service is deployed and accessible
- Check the base URL in the test file matches your deployment

## 📊 Test Results

### Success Criteria

- **FastAPI Subprocess Fix**: 100% pass rate (all 3 tests)
- **Cloud Run API**: 80%+ pass rate

### What to Do If Tests Fail

1. **Check the logs** for specific error messages
2. **Verify environment setup** (variables, binaries, permissions)
3. **Run tests individually** to isolate issues
4. **Check recent changes** that might have affected the functionality

## 🧹 Cleanup History

The following test files were removed during cleanup as they were redundant or outdated:

- `test_video_download.py` - Superseded by FastAPI subprocess fix
- `test_multiple_videos.py` - Redundant functionality
- `quick_multi_test.py` - Simple test, not comprehensive
- `test_quality_fallback.py` - Outdated with current architecture
- `test_updated_system.py` - Vague purpose, superseded
- `test_enhanced_downloader.py` - Covered by other tests
- `test_stuck_download_recovery.py` - Issue resolved with subprocess fix
- `test_fixed_downloader.py` - Outdated
- `test_real_video_download.py` - Redundant
- `test_full_raw_pipeline.py` - Overlapped with other tests
- `demo_quality_fallback.py` - Demo script, not essential
- `validate_implementation.py` - One-time validation, superseded

## 📝 Adding New Tests

When adding new test files:

1. **Use descriptive names** that clearly indicate the test purpose
2. **Include comprehensive documentation** in the file header
3. **Follow the existing test patterns** for consistency
4. **Update this documentation** to include the new test
5. **Ensure tests are independent** and can run in any order

## 🎯 Best Practices

- **Run tests before committing** changes
- **Keep tests focused** on specific functionality
- **Use meaningful assertions** with clear error messages
- **Clean up resources** after tests complete
- **Document expected behavior** and failure scenarios
