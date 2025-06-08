# Parakeet Transcriber Implementation

## Overview

This document describes the implementation of an alternative transcription module using the NVIDIA Parakeet TDT 0.6B v2 model as a local, cost-effective replacement for the Deepgram transcription service.

## Implementation Details

### Files Created

1. **`raw_pipeline/transcriber_parakeet.py`** - Main Parakeet transcriber implementation
2. **`test_transcriber_comparison.py`** - Comparison test script between Deepgram and Parakeet
3. **`test_parakeet_integration.py`** - Integration test for drop-in replacement functionality
4. **`test_audio_conversion.py`** - Audio format conversion testing script

### Key Features

#### 1. Drop-in Replacement Compatibility
- **Identical Interface**: Same class structure and method signatures as `TranscriptionHandler`
- **Same Input Parameters**: `video_id`, `audio_file_path`, `output_dir`
- **Same Output Format**: Generates identical CSV and JSON files with same column names
- **Same Error Handling**: Maintains logging patterns and error recovery

#### 2. Audio Format Conversion
- **Automatic MP3 to WAV Conversion**: Handles MP3 input files by converting to WAV
- **Multiple Format Support**: Supports MP3, M4A, AAC, OGG → WAV/FLAC conversion
- **Format Validation**: Checks and validates audio formats before processing
- **Temporary File Management**: Cleans up converted files automatically

#### 3. Model Optimization
- **GPU/CPU Fallback**: Automatically detects and uses best available device
- **Model Caching**: Loads model once and reuses for multiple transcriptions
- **Memory Optimization**: Uses float16 on GPU, float32 on CPU
- **Chunking Strategy**: Processes long audio files in 30-second chunks

#### 4. Audio Preprocessing
- **Sample Rate Normalization**: Converts all audio to 16kHz
- **Mono Conversion**: Converts stereo to mono
- **Audio Validation**: Checks for NaN, infinite values, and proper ranges
- **Normalization**: Ensures audio values are in [-1, 1] range

### Technical Specifications

#### Model Details
- **Model**: `nvidia/parakeet-tdt-0.6b-v2`
- **Type**: Speech-to-Text Transformer
- **Input**: 16kHz mono audio
- **Output**: Text transcription with timestamps

#### Audio Requirements
- **Sample Rate**: 16kHz (automatically converted)
- **Channels**: Mono (automatically converted from stereo)
- **Format**: WAV or FLAC (MP3 automatically converted)
- **Bit Depth**: 16-bit or 32-bit float

#### Performance Characteristics
- **Chunk Size**: 30 seconds (configurable)
- **Memory Usage**: ~2-4GB GPU memory for model
- **Processing Speed**: Varies by hardware (typically 2-10x real-time)

### Dependencies Added

```python
# Hugging Face and ML models for local transcription
transformers>=4.36.0
torch>=2.1.0
torchaudio>=2.1.0
accelerate>=0.25.0
```

### Usage Examples

#### Basic Usage (Drop-in Replacement)
```python
from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler

# Replace this:
# transcriber = TranscriptionHandler()

# With this:
transcriber = ParakeetTranscriptionHandler()

# Same interface
result = await transcriber.process_audio_files(
    video_id="12345",
    audio_file_path="/path/to/audio.mp3"
)
```

#### Running Comparison Tests
```bash
# Compare both transcribers
python test_transcriber_comparison.py 12345 --audio-file /path/to/audio.mp3

# Test integration
python test_parakeet_integration.py 12345 --audio-file /path/to/audio.mp3

# Test audio conversion
python test_audio_conversion.py /path/to/audio.mp3
```

### Output Compatibility

The Parakeet transcriber generates identical output files:

#### Words CSV (`audio_{video_id}_words.csv`)
```csv
start_time,end_time,word
0.0,0.5,Hello
0.5,1.0,world
```

#### Paragraphs CSV (`audio_{video_id}_paragraphs.csv`)
```csv
start_time,end_time,text
0.0,60.0,Hello world this is a paragraph...
```

#### JSON Transcript (`audio_{video_id}_transcript.json`)
```json
{
  "results": {
    "channels": [{
      "alternatives": [{
        "transcript": "Hello world...",
        "words": [...]
      }]
    }]
  },
  "metadata": {
    "model": "nvidia/parakeet-tdt-0.6b-v2",
    "duration": 120.5,
    "channels": 1
  }
}
```

### Integration with Existing Pipeline

To use Parakeet instead of Deepgram in the main pipeline:

1. **Option 1: Direct Replacement**
   ```python
   # In raw_pipeline/processor.py, line 101:
   # from .transcriber import TranscriptionHandler
   from .transcriber_parakeet import ParakeetTranscriptionHandler as TranscriptionHandler
   ```

2. **Option 2: Configuration-Based**
   ```python
   # Add to utils/config.py
   USE_LOCAL_TRANSCRIPTION = os.getenv("USE_LOCAL_TRANSCRIPTION", "false").lower() == "true"
   
   # In processor.py
   if USE_LOCAL_TRANSCRIPTION:
       from .transcriber_parakeet import ParakeetTranscriptionHandler as TranscriptionHandler
   else:
       from .transcriber import TranscriptionHandler
   ```

### Performance Considerations

#### Advantages
- **Cost**: No API costs after initial setup
- **Privacy**: All processing happens locally
- **Offline**: Works without internet connection
- **Consistency**: Deterministic results

#### Disadvantages
- **Speed**: May be slower than Deepgram API (depends on hardware)
- **Hardware**: Requires GPU for optimal performance
- **Model Size**: ~2.4GB model download
- **Memory**: Higher memory usage than API calls

### Testing Strategy

#### 1. Unit Tests
- Audio format conversion
- Model loading and initialization
- Chunking and processing logic

#### 2. Integration Tests
- Drop-in replacement functionality
- Output format compatibility
- Error handling consistency

#### 3. Performance Tests
- Processing speed comparison
- Memory usage monitoring
- Quality assessment (WER, similarity metrics)

#### 4. End-to-End Tests
- Full pipeline integration
- Large file processing
- Error recovery scenarios

### Quality Assessment

The comparison script provides several metrics:

- **Processing Time**: Deepgram vs Parakeet speed
- **Word Count**: Number of words transcribed
- **Character Similarity**: Text overlap between outputs
- **Word Similarity**: Vocabulary overlap
- **File Sizes**: Output file size comparison

### Troubleshooting

#### Common Issues

1. **CUDA Out of Memory**
   - Reduce chunk size
   - Use CPU instead of GPU
   - Close other GPU applications

2. **Audio Format Errors**
   - Check input file format
   - Verify pydub installation
   - Test with `test_audio_conversion.py`

3. **Model Loading Errors**
   - Check internet connection for initial download
   - Verify transformers library version
   - Clear Hugging Face cache if corrupted

4. **Slow Performance**
   - Use GPU if available
   - Reduce chunk size for memory constraints
   - Consider model quantization

### Future Improvements

1. **Model Quantization**: Reduce memory usage with INT8 quantization
2. **Batch Processing**: Process multiple chunks simultaneously
3. **Streaming**: Real-time transcription for live audio
4. **Fine-tuning**: Adapt model for specific domains/accents
5. **Alternative Models**: Support for other local speech models

### Conclusion

The Parakeet transcriber provides a viable local alternative to Deepgram with:
- Full compatibility with existing pipeline
- Automatic audio format handling
- Comprehensive testing suite
- Performance monitoring capabilities

This implementation enables cost optimization and offline capability while maintaining the same output quality and format as the original Deepgram implementation.
