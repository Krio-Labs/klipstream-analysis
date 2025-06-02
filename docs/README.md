# 📚 KlipStream Analysis Documentation

Welcome to the comprehensive documentation for KlipStream Analysis - a production-ready API for analyzing Twitch VODs.

## 🚀 **Getting Started**

### For Developers
- **[📖 API Documentation](API_DOCUMENTATION.md)** - Complete API reference and usage guide
- **[⚛️ Next.js Integration](NEXTJS_INTEGRATION.md)** - Frontend integration with React components

### For DevOps/Deployment
- **[🚀 Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment and maintenance

## 📋 **Documentation Overview**

### 📖 **API Documentation**
Complete reference for the KlipStream Analysis API including:
- Endpoint specifications
- Request/response formats
- Error handling
- Processing pipeline details
- File storage structure
- Performance characteristics

### ⚛️ **Next.js Integration Guide**
Step-by-step guide for integrating the API into Next.js applications:
- TypeScript API service layer
- React hooks for state management
- Complete UI components
- Error handling and loading states
- Best practices and testing

### 🚀 **Deployment Guide**
Comprehensive deployment documentation covering:
- Google Cloud Run setup
- Environment configuration
- Monitoring and maintenance
- Security configuration
- Troubleshooting guide

## 🎯 **Quick Reference**

### Production API
```
Base URL: https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app
Method: POST
Content-Type: application/json
Body: {"url": "https://www.twitch.tv/videos/VIDEO_ID"}
```

### Processing Status Flow
```
Queued → Downloading → Fetching chat → Transcribing → 
Analyzing → Finding highlights → Completed
```

### Typical Processing Times
- **1-hour VOD**: 5-7 minutes
- **2-hour VOD**: 8-12 minutes
- **30-minute VOD**: 3-5 minutes

## 🔗 **External Resources**

- **[Main Repository](https://github.com/Krio-Labs/klipstream-analysis)** - Source code and technical documentation
- **[Production API](https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app)** - Live API endpoint
- **[Google Cloud Run](https://cloud.google.com/run)** - Deployment platform

## 📞 **Support**

- **Issues**: Report bugs or request features via GitHub Issues
- **Integration Help**: Contact the development team for API integration support
- **Documentation**: Suggest improvements or report errors in documentation

---

Choose the documentation that best fits your needs and get started with KlipStream Analysis! 🚀
