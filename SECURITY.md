# Security Guidelines

## Environment Variables and Secrets Management

### ⚠️ IMPORTANT: Never commit secrets to version control

This project uses several API keys and sensitive configuration values that must be kept secure:

- `DEEPGRAM_API_KEY` - Deepgram transcription service
- `NEBIUS_API_KEY` - Nebius sentiment analysis service
- `GOOGLE_API_KEY` - Google Cloud services
- `CONVEX_API_KEY` - Convex database access
- `GCP_SERVICE_ACCOUNT_PATH` - Google Cloud service account key file

### Local Development Setup

1. **Copy the example files:**
   ```bash
   cp .env.example .env
   cp .env.yaml.example .env.yaml
   ```

2. **Fill in your actual API keys** in the `.env` and `.env.yaml` files

3. **Use direnv for automatic environment loading:**
   ```bash
   # Install direnv (macOS)
   brew install direnv
   
   # Add to your shell (choose one)
   echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
   echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
   
   # Allow direnv in this project
   direnv allow
   ```

### Cloud Deployment

For Cloud Run deployment, environment variables are set directly in the deployment script and not stored in files.

### Files to Keep Secure

**Never commit these files:**
- `.env` - Contains actual API keys
- `.env.yaml` - Contains actual API keys  
- `*-service-account-key.json` - Google Cloud service account keys

**Safe to commit:**
- `.envrc` - Direnv configuration (no secrets)
- `.env.example` - Template with placeholder values
- `.env.yaml.example` - Template with placeholder values

### Checking for Exposed Secrets

Before committing, always check:
```bash
# Check what files are staged
git status

# Make sure no .env files are tracked
git ls-files | grep -E '\.(env|yaml)$'

# Should only show example files, not actual .env/.env.yaml
```

### If You Accidentally Commit Secrets

1. **Immediately rotate all exposed API keys**
2. **Remove the secrets from git history:**
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch .env .env.yaml' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push to update remote repository**
4. **Update all deployment environments with new keys**

### Additional Security Measures

- Use least-privilege access for service accounts
- Regularly rotate API keys
- Monitor API usage for unusual activity
- Use environment-specific configurations
- Enable audit logging where available
