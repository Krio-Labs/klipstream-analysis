# 🚀 Production Deployment Guide

This guide covers the complete production deployment process for both the FastAPI backend and Next.js frontend of the KlipStream Analysis system.

## 📋 Pre-Deployment Checklist

### ✅ Backend Readiness
- [ ] All Phase 4 tests passing (100% success rate achieved)
- [ ] Load testing completed (6,000+ RPS validated)
- [ ] Environment variables configured
- [ ] Service account keys secured
- [ ] Database connections tested
- [ ] Storage buckets accessible
- [ ] Monitoring and alerting configured

### ✅ Frontend Readiness
- [ ] API client implemented and tested
- [ ] Real-time features working
- [ ] Error handling implemented
- [ ] State management configured
- [ ] Performance optimized
- [ ] Mobile responsiveness verified

### ✅ Infrastructure Readiness
- [ ] Google Cloud Project configured
- [ ] Cloud Run services provisioned
- [ ] Domain names configured
- [ ] SSL certificates ready
- [ ] CDN configured (if needed)
- [ ] Monitoring tools set up

## 🏗️ Backend Deployment (FastAPI to Cloud Run)

### Step 1: Final Backend Testing

First, let's run our comprehensive test suite to ensure production readiness:

```bash
# Run Phase 4 deployment tests
cd /path/to/klipstream-analysis
python test_phase4_deployment.py

# Run load testing
python test_load_performance.py

# Verify all services are healthy
curl http://localhost:3001/health
curl http://localhost:3001/api/v1/monitoring/health
```

### Step 2: Environment Configuration

Create production environment file `.env.production`:

```bash
# Production Environment Variables
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# API Configuration
API_VERSION=v1
MAX_CONCURRENT_JOBS=5
REQUEST_TIMEOUT=300
CORS_ORIGINS=https://your-frontend-domain.com

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=klipstream
GCS_BUCKET_VODS=klipstream-vods-raw
GCS_BUCKET_TRANSCRIPTS=klipstream-transcripts
GCS_BUCKET_CHATLOGS=klipstream-chatlogs
GCS_BUCKET_ANALYSIS=klipstream-analysis

# Database Configuration
CONVEX_URL=https://laudable-horse-446.convex.cloud
CONVEX_API_KEY=your-production-api-key

# External Services
DEEPGRAM_API_KEY=your-production-deepgram-key
NEBIUS_API_KEY=your-production-nebius-key

# Monitoring
ENABLE_METRICS=true
METRICS_RETENTION_HOURS=24
ALERT_WEBHOOK_URL=https://your-monitoring-webhook.com

# Performance Tuning
UVICORN_WORKERS=2
UVICORN_TIMEOUT_KEEP_ALIVE=65
UVICORN_MAX_REQUESTS=1000
UVICORN_MAX_REQUESTS_JITTER=100
```

### Step 3: Production Dockerfile Optimization

Update your Dockerfile for production:

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy application code
COPY --chown=app:app . .

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONPATH=/home/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Production command with optimizations
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout-keep-alive", "65", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-log", \
     "--log-level", "info"]
```

### Step 4: Enhanced Deployment Script

Update `deploy_cloud_run_simple.sh` for production:

```bash
#!/bin/bash

set -e

# Production deployment configuration
PROJECT_ID="klipstream"
SERVICE_NAME="klipstream-analysis"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
ENVIRONMENT="production"

echo "🚀 Starting Production Deployment to Cloud Run"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Environment: ${ENVIRONMENT}"
echo ""

# Authenticate with Google Cloud
echo "🔐 Authenticating with Google Cloud..."
gcloud auth configure-docker --quiet

# Build production image
echo "🏗️ Building production Docker image..."
docker build -t ${IMAGE_NAME}:latest -t ${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S) .

# Push image to Container Registry
echo "📤 Pushing image to Container Registry..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run with production settings
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --allow-unauthenticated \
    --memory=4Gi \
    --cpu=2 \
    --timeout=900 \
    --concurrency=100 \
    --max-instances=10 \
    --min-instances=1 \
    --port=8080 \
    --set-env-vars="ENVIRONMENT=${ENVIRONMENT}" \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-env-vars="GCS_BUCKET_VODS=klipstream-vods-raw" \
    --set-env-vars="GCS_BUCKET_TRANSCRIPTS=klipstream-transcripts" \
    --set-env-vars="GCS_BUCKET_CHATLOGS=klipstream-chatlogs" \
    --set-env-vars="GCS_BUCKET_ANALYSIS=klipstream-analysis" \
    --quiet

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""

# Run production health checks
echo "🏥 Running production health checks..."
echo "Testing basic health endpoint..."
curl -f "${SERVICE_URL}/health" || exit 1

echo "Testing API documentation..."
curl -f "${SERVICE_URL}/docs" || exit 1

echo "Testing monitoring endpoint..."
curl -f "${SERVICE_URL}/api/v1/monitoring/health" || exit 1

echo ""
echo "🎉 Production deployment successful!"
echo "📊 Monitor at: ${SERVICE_URL}/api/v1/monitoring/dashboard"
echo "📖 API Docs at: ${SERVICE_URL}/docs"
```

### Step 5: Deploy Backend

Execute the production deployment:

```bash
# Make script executable
chmod +x deploy_cloud_run_simple.sh

# Deploy to production
./deploy_cloud_run_simple.sh
```

## 🌐 Frontend Deployment (Next.js)

### Step 1: Production Environment Setup

Create `.env.production` for your Next.js app:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
NEXT_PUBLIC_API_VERSION=v1

# Feature Flags
NEXT_PUBLIC_ENABLE_REAL_TIME_UPDATES=true
NEXT_PUBLIC_ENABLE_MONITORING_DASHBOARD=true
NEXT_PUBLIC_ENABLE_QUEUE_MANAGEMENT=true

# WebSocket/SSE Configuration
NEXT_PUBLIC_WS_URL=wss://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
NEXT_PUBLIC_SSE_RECONNECT_INTERVAL=5000

# Auth Configuration
NEXT_PUBLIC_AUTH0_DOMAIN=dev-6umplkv2jurpmp1m.us.auth0.com
NEXT_PUBLIC_AUTH0_CLIENT_ID=bXyThTPq0KD5WlzHp2OBG7fHX2RIU7Ob

# Analytics (optional)
NEXT_PUBLIC_GA_TRACKING_ID=your-ga-tracking-id
NEXT_PUBLIC_SENTRY_DSN=your-sentry-dsn

# Performance
NEXT_PUBLIC_CACHE_TTL=300000
NEXT_PUBLIC_MAX_RETRIES=3
```

### Step 2: Production Build Optimization

Update `next.config.js` for production:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Production optimizations
  compress: true,
  poweredByHeader: false,
  
  // Image optimization
  images: {
    domains: ['storage.googleapis.com'],
    formats: ['image/webp', 'image/avif'],
  },
  
  // Headers for security and performance
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },
  
  // Redirects for API endpoints
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL}/api/v1/:path*`,
      },
    ];
  },
  
  // Bundle analyzer (development only)
  ...(process.env.ANALYZE === 'true' && {
    webpack: (config) => {
      config.plugins.push(
        new (require('@next/bundle-analyzer'))({
          enabled: true,
        })
      );
      return config;
    },
  }),
};

module.exports = nextConfig;
```

### Step 3: Deployment Options

Choose your preferred deployment platform:

#### Option A: Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to production
vercel --prod

# Set environment variables
vercel env add NEXT_PUBLIC_API_URL production
vercel env add NEXT_PUBLIC_AUTH0_DOMAIN production
# ... add all other environment variables
```

#### Option B: Google Cloud Run

Create `Dockerfile` for Next.js:

```dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --only=production

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED 1
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
```

Deploy to Cloud Run:

```bash
# Build and deploy
gcloud run deploy klipstream-frontend \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 5
```

#### Option C: Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Build and deploy
npm run build
netlify deploy --prod --dir=.next
```

## 🔒 Security Configuration

### Backend Security

1. **Service Account Security**:
```bash
# Create production service account
gcloud iam service-accounts create klipstream-prod \
    --display-name="KlipStream Production Service Account"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding klipstream \
    --member="serviceAccount:klipstream-prod@klipstream.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

2. **API Security**:
```python
# Add to api/main.py
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["klipstream-analysis-4vyl5ph7lq-uc.a.run.app", "*.klipstream.com"]
)
```

### Frontend Security

1. **Content Security Policy**:
```javascript
// Add to next.config.js
const securityHeaders = [
  {
    key: 'Content-Security-Policy',
    value: `
      default-src 'self';
      script-src 'self' 'unsafe-eval' 'unsafe-inline';
      style-src 'self' 'unsafe-inline';
      img-src 'self' data: https:;
      connect-src 'self' ${process.env.NEXT_PUBLIC_API_URL};
    `.replace(/\s{2,}/g, ' ').trim()
  }
];
```

2. **Environment Variable Validation**:
```typescript
// Add to lib/config.ts
const requiredEnvVars = [
  'NEXT_PUBLIC_API_URL',
  'NEXT_PUBLIC_AUTH0_DOMAIN',
  'NEXT_PUBLIC_AUTH0_CLIENT_ID',
];

requiredEnvVars.forEach((envVar) => {
  if (!process.env[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}`);
  }
});
```

## 📊 Monitoring & Observability

### Backend Monitoring

1. **Google Cloud Monitoring**:
```bash
# Enable monitoring APIs
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

2. **Custom Metrics Dashboard**:
```python
# Add to monitoring setup
from google.cloud import monitoring_v3

def setup_custom_metrics():
    client = monitoring_v3.MetricServiceClient()
    # Configure custom metrics for job processing, error rates, etc.
```

### Frontend Monitoring

1. **Error Tracking with Sentry**:
```bash
npm install @sentry/nextjs
```

```javascript
// sentry.client.config.js
import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV,
});
```

2. **Performance Monitoring**:
```typescript
// lib/analytics.ts
export const trackEvent = (eventName: string, properties: Record<string, any>) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', eventName, properties);
  }
};
```

## 🚀 Deployment Automation

### CI/CD Pipeline with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python test_phase4_deployment.py

  deploy-backend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: google-github-actions/setup-gcloud@v1
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: klipstream
      - name: Deploy to Cloud Run
        run: ./deploy_cloud_run_simple.sh

  deploy-frontend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm ci
      - name: Build
        run: npm run build
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

## ✅ Post-Deployment Verification

### Automated Health Checks

Create `scripts/production-health-check.sh`:

```bash
#!/bin/bash

API_URL="https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"
FRONTEND_URL="https://your-frontend-domain.com"

echo "🏥 Running Production Health Checks"
echo "=================================="

# Backend health checks
echo "Testing backend health..."
curl -f "${API_URL}/health" || exit 1

echo "Testing API endpoints..."
curl -f "${API_URL}/api/v1/monitoring/health" || exit 1

echo "Testing queue status..."
curl -f "${API_URL}/api/v1/queue/status" || exit 1

# Frontend health checks
echo "Testing frontend..."
curl -f "${FRONTEND_URL}" || exit 1

echo "Testing API integration..."
curl -f "${FRONTEND_URL}/api/health" || exit 1

echo "✅ All health checks passed!"
```

### Performance Validation

```bash
# Load testing in production
ab -n 1000 -c 10 https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/health

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app/health
```

## 🎯 Success Metrics

Monitor these key metrics post-deployment:

### Backend Metrics:
- ✅ **Response Time**: <2 seconds for job initiation
- ✅ **Throughput**: 1000+ requests/minute
- ✅ **Error Rate**: <1%
- ✅ **Uptime**: 99.9%

### Frontend Metrics:
- ✅ **Page Load Time**: <3 seconds
- ✅ **Time to Interactive**: <5 seconds
- ✅ **Core Web Vitals**: All green
- ✅ **Error Rate**: <0.5%

## 🔄 Blue-Green Deployment Strategy

### Backend Blue-Green Deployment

Implement zero-downtime deployments:

```bash
#!/bin/bash
# blue-green-deploy.sh

SERVICE_NAME="klipstream-analysis"
CURRENT_SERVICE="${SERVICE_NAME}-blue"
NEW_SERVICE="${SERVICE_NAME}-green"

# Deploy to green environment
echo "🟢 Deploying to green environment..."
gcloud run deploy ${NEW_SERVICE} \
    --image=${IMAGE_NAME}:latest \
    --platform=managed \
    --region=${REGION} \
    --no-traffic

# Health check green environment
echo "🏥 Health checking green environment..."
GREEN_URL=$(gcloud run services describe ${NEW_SERVICE} --format="value(status.url)")
curl -f "${GREEN_URL}/health" || exit 1

# Gradually shift traffic
echo "🔄 Shifting traffic to green..."
gcloud run services update-traffic ${SERVICE_NAME} \
    --to-revisions=${NEW_SERVICE}=10,${CURRENT_SERVICE}=90

# Monitor for 5 minutes
sleep 300

# Complete traffic shift if healthy
gcloud run services update-traffic ${SERVICE_NAME} \
    --to-revisions=${NEW_SERVICE}=100

echo "✅ Blue-green deployment completed!"
```

### Frontend Blue-Green with Vercel

```bash
# Deploy to preview environment first
vercel --target preview

# Run integration tests against preview
npm run test:integration -- --baseUrl=$PREVIEW_URL

# Promote to production if tests pass
vercel --prod
```

## 🔧 Advanced Configuration

### Database Connection Pooling

```python
# Add to api/services/database.py
import asyncpg
from asyncpg import Pool

class DatabaseManager:
    def __init__(self):
        self.pool: Pool = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            min_size=5,
            max_size=20,
            command_timeout=60,
        )

    async def close(self):
        if self.pool:
            await self.pool.close()

db_manager = DatabaseManager()
```

### Redis Caching Layer

```python
# Add to api/services/cache.py
import redis.asyncio as redis
from typing import Optional, Any
import json

class RedisCache:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None

    async def initialize(self):
        self.redis = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )

    async def get(self, key: str) -> Optional[Any]:
        if not self.redis:
            return None

        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        if self.redis:
            await self.redis.setex(key, ttl, json.dumps(value))

cache = RedisCache()
```

### Load Balancer Configuration

```yaml
# cloud-run-load-balancer.yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: klipstream-ssl-cert
spec:
  domains:
    - api.klipstream.com
---
apiVersion: networking.gke.io/v1beta1
kind: BackendConfig
metadata:
  name: klipstream-backend-config
spec:
  healthCheck:
    checkIntervalSec: 30
    timeoutSec: 10
    healthyThreshold: 1
    unhealthyThreshold: 3
    type: HTTP
    requestPath: /health
  connectionDraining:
    drainingTimeoutSec: 60
```

## 📈 Scaling Configuration

### Auto-scaling Setup

```bash
# Configure Cloud Run auto-scaling
gcloud run services update klipstream-analysis \
    --min-instances=2 \
    --max-instances=50 \
    --concurrency=100 \
    --cpu-throttling \
    --memory=4Gi \
    --cpu=2
```

### Horizontal Pod Autoscaler (if using GKE)

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: klipstream-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: klipstream-analysis
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🛡️ Advanced Security

### WAF Configuration

```yaml
# cloud-armor-policy.yaml
apiVersion: compute/v1
kind: SecurityPolicy
metadata:
  name: klipstream-security-policy
spec:
  rules:
  - priority: 1000
    match:
      versionedExpr: SRC_IPS_V1
      config:
        srcIpRanges:
        - "192.168.1.0/24"  # Your office IP range
    action: allow
  - priority: 2000
    match:
      expr:
        expression: 'origin.region_code == "CN"'
    action: deny(403)
  - priority: 2147483647
    match:
      versionedExpr: SRC_IPS_V1
      config:
        srcIpRanges:
        - "*"
    action: allow
```

### API Rate Limiting

```python
# Add to api/middleware/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/analysis")
@limiter.limit("10/minute")
async def start_analysis(request: Request, ...):
    # Analysis endpoint with rate limiting
    pass
```

## 📊 Advanced Monitoring

### Custom Metrics Dashboard

```python
# Add to api/services/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Custom metrics
job_counter = Counter('analysis_jobs_total', 'Total analysis jobs', ['status'])
job_duration = Histogram('analysis_job_duration_seconds', 'Job duration')
active_jobs = Gauge('analysis_jobs_active', 'Currently active jobs')

class MetricsCollector:
    def record_job_start(self):
        active_jobs.inc()
        return time.time()

    def record_job_completion(self, start_time: float, status: str):
        duration = time.time() - start_time
        job_duration.observe(duration)
        job_counter.labels(status=status).inc()
        active_jobs.dec()
```

### Alerting Rules

```yaml
# alerting-rules.yaml
groups:
- name: klipstream-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighMemoryUsage
    expr: memory_usage_percent > 90
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High memory usage
      description: "Memory usage is {{ $value }}%"
```

## 🔄 Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup-strategy.sh

# Database backup
gcloud sql export sql klipstream-db gs://klipstream-backups/db-backup-$(date +%Y%m%d).sql

# Storage backup
gsutil -m cp -r gs://klipstream-vods-raw gs://klipstream-backups/vods-$(date +%Y%m%d)/

# Configuration backup
kubectl get configmaps -o yaml > configmaps-backup-$(date +%Y%m%d).yaml
```

### Recovery Procedures

```bash
#!/bin/bash
# disaster-recovery.sh

echo "🚨 Starting disaster recovery procedure..."

# 1. Restore database
gcloud sql import sql klipstream-db gs://klipstream-backups/db-backup-latest.sql

# 2. Restore storage
gsutil -m cp -r gs://klipstream-backups/vods-latest/ gs://klipstream-vods-raw/

# 3. Redeploy services
./deploy_cloud_run_simple.sh

# 4. Verify health
./scripts/production-health-check.sh

echo "✅ Disaster recovery completed!"
```

## 🎯 Performance Optimization

### CDN Configuration

```bash
# Configure Cloud CDN
gcloud compute backend-services update klipstream-backend \
    --enable-cdn \
    --cache-mode=CACHE_ALL_STATIC \
    --default-ttl=3600 \
    --max-ttl=86400
```

### Database Optimization

```sql
-- Add database indexes for performance
CREATE INDEX CONCURRENTLY idx_videos_status ON videos(status);
CREATE INDEX CONCURRENTLY idx_videos_created_at ON videos(created_at);
CREATE INDEX CONCURRENTLY idx_jobs_video_id ON analysis_jobs(video_id);
```

### Frontend Performance

```javascript
// next.config.js - Advanced optimizations
module.exports = {
  experimental: {
    optimizeCss: true,
    optimizeImages: true,
    optimizeServerReact: true,
  },

  // Bundle splitting
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.optimization.splitChunks.cacheGroups = {
        ...config.optimization.splitChunks.cacheGroups,
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
        },
      };
    }
    return config;
  },
};
```

## 📋 Production Checklist

### Pre-Launch Checklist
- [ ] Load testing completed (>1000 RPS)
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Monitoring dashboards configured
- [ ] Alerting rules tested
- [ ] Backup procedures verified
- [ ] Disaster recovery plan tested
- [ ] Documentation updated
- [ ] Team training completed

### Post-Launch Monitoring
- [ ] Error rates < 1%
- [ ] Response times < 2s
- [ ] Uptime > 99.9%
- [ ] Memory usage < 80%
- [ ] CPU usage < 70%
- [ ] Disk usage < 85%
- [ ] Queue processing healthy
- [ ] Real-time updates working

Your production deployment is now complete with enterprise-grade reliability! 🎉
