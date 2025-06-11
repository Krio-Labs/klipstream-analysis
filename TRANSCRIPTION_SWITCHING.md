# 🎤 Transcription Method Switching Guide

## 🔄 **Temporary Deepgram Usage**

The klipstream-analysis pipeline is **temporarily using Deepgram** for all transcription while GPU transcription is being optimized. This is a temporary measure that can be easily reverted.

## 🚀 **Quick Switching**

### **Option 1: Using the Switch Script (Recommended)**
```bash
# Switch to Deepgram (current default)
python switch_transcription.py deepgram

# Switch to GPU transcription
python switch_transcription.py gpu

# Check current status
python switch_transcription.py status

# Toggle between methods
python switch_transcription.py toggle
```

### **Option 2: Using Environment Variable**
```bash
# Use Deepgram (temporary)
export FORCE_DEEPGRAM_TRANSCRIPTION=true
python main.py

# Use GPU transcription (original)
export FORCE_DEEPGRAM_TRANSCRIPTION=false
python main.py
```

### **Option 3: Edit Configuration File**
Edit `transcription_config.py` and change:
```python
# For Deepgram (temporary)
USE_DEEPGRAM_TEMPORARILY = True

# For GPU transcription (original)
USE_DEEPGRAM_TEMPORARILY = False
```

## 📊 **Current Configuration Status**

Run this to see the current transcription configuration:
```bash
python switch_transcription.py status
```

Example output:
```
🎤 TRANSCRIPTION CONFIGURATION STATUS
==================================================
Current method: deepgram
Using Deepgram: True
Using GPU: False
Temporary override: True
Environment override: False
Config source: module_constant

💡 TO RE-ENABLE GPU TRANSCRIPTION:
   Option 1: Set environment variable FORCE_DEEPGRAM_TRANSCRIPTION=false
   Option 2: Change USE_DEEPGRAM_TEMPORARILY = False in transcription_config.py
   Option 3: Call set_transcription_method(False)
==================================================
```

## 🌐 **Deepgram Configuration**

When using Deepgram, make sure you have:

1. **API Key**: Set `DEEPGRAM_API_KEY` environment variable
2. **Model**: Uses `nova-3` model by default
3. **Features**: Enabled smart_format, paragraphs, punctuate, filler_words

```bash
export DEEPGRAM_API_KEY="your_deepgram_api_key_here"
```

## 🖥️ **GPU Transcription Configuration**

When using GPU transcription:

1. **Hardware**: Requires NVIDIA GPU with CUDA or Apple Silicon with Metal
2. **Model**: Uses NVIDIA Parakeet TDT 0.6B v2 by default
3. **Memory**: Requires 2GB+ GPU memory
4. **Dependencies**: NeMo toolkit and compatible PyTorch

## 🔧 **Integration with Main Pipeline**

The transcription method is automatically detected and applied in `main.py`:

```python
# Configure transcription environment
transcription_config, gpu_info = configure_transcription_environment()

# Logs will show:
# 🎤 Transcription method: deepgram (TEMPORARY - GPU transcription disabled)
# 💡 To re-enable GPU transcription: python transcription_config.py gpu
```

## 📝 **Files Involved**

### **Core Files**
- `main.py` - Main pipeline with transcription configuration
- `transcription_config.py` - Configuration management module
- `switch_transcription.py` - Easy switching script

### **Transcription Handlers**
- `raw_pipeline/transcription/handlers/deepgram_handler.py` - Deepgram implementation
- `raw_pipeline/transcription/handlers/parakeet_gpu.py` - GPU implementation
- `raw_pipeline/transcription/transcriber.py` - Main transcription orchestrator

## 🎯 **When to Use Each Method**

### **Use Deepgram When:**
- ✅ Need reliable, consistent transcription
- ✅ Don't have compatible GPU hardware
- ✅ Want to avoid GPU memory issues
- ✅ Processing occasional videos (cost-effective for low volume)

### **Use GPU Transcription When:**
- ✅ Have compatible NVIDIA GPU or Apple Silicon
- ✅ Processing high volume of videos (cost-effective)
- ✅ Want offline transcription capability
- ✅ GPU optimization issues have been resolved

## 🔄 **Switching Back to GPU Transcription**

When ready to switch back to GPU transcription:

```bash
# Method 1: Use the switch script
python switch_transcription.py gpu

# Method 2: Set environment variable
export FORCE_DEEPGRAM_TRANSCRIPTION=false

# Method 3: Edit transcription_config.py
# Change USE_DEEPGRAM_TEMPORARILY = False
```

## 🚨 **Important Notes**

### **Temporary Nature**
- This Deepgram usage is **temporary** while GPU transcription is optimized
- The original GPU transcription setup is preserved and ready to re-enable
- No breaking changes to the pipeline architecture

### **Cost Considerations**
- **Deepgram**: Pay per minute of audio transcribed
- **GPU**: One-time setup cost, then free for unlimited transcription
- Monitor usage and costs when using Deepgram

### **Performance**
- **Deepgram**: Consistent cloud-based performance
- **GPU**: Performance depends on local hardware capabilities

## 🔍 **Troubleshooting**

### **Deepgram Issues**
```bash
# Check API key
echo $DEEPGRAM_API_KEY

# Test Deepgram connection
python -c "from raw_pipeline.transcription.handlers.deepgram_handler import DeepgramHandler; print('Deepgram OK')"
```

### **GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check NeMo installation
python -c "import nemo; print('NeMo OK')"
```

### **Configuration Issues**
```bash
# Reset to default (Deepgram)
python switch_transcription.py deepgram

# Check current status
python switch_transcription.py status
```

## 📞 **Support**

If you encounter issues with transcription switching:

1. **Check current status**: `python switch_transcription.py status`
2. **Reset to Deepgram**: `python switch_transcription.py deepgram`
3. **Check logs**: Look for transcription method in pipeline logs
4. **Environment**: Verify `DEEPGRAM_API_KEY` is set for Deepgram usage

---

**Status**: ✅ **Temporary Deepgram usage active** - GPU transcription preserved for easy re-enabling
