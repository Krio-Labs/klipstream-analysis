# Migration from AssemblyAI to Deepgram

This document outlines the migration from AssemblyAI to Deepgram for speech-to-text transcription in the Klipstream Analysis project.

## Motivation

The decision to migrate from AssemblyAI to Deepgram was made for the following reasons:

1. **Cost Efficiency**: Deepgram offers a more cost-effective pricing model for our usage patterns.
2. **Performance**: Deepgram provides comparable or better transcription accuracy with faster processing times.
3. **Feature Parity**: Deepgram offers similar features to AssemblyAI, including word-level timestamps and paragraph detection.
4. **API Simplicity**: Deepgram's API is more straightforward and requires fewer API calls to get both word-level and paragraph-level transcriptions.

## Implementation Changes

The following changes were made to implement the migration:

1. **Configuration Updates**:
   - Added `DEEPGRAM_API_KEY` to `.env.yaml`
   - Updated `utils/config.py` to load the Deepgram API key

2. **Dependencies**:
   - Added `deepgram-sdk==3.1.0` to `requirements.txt`

3. **Transcriber Module**:
   - Rewrote `raw_pipeline/transcriber.py` to use Deepgram instead of AssemblyAI
   - Implemented equivalent functionality for word-level timestamps and paragraph detection
   - Added fallback logic to create paragraphs from sentences if Deepgram's paragraph feature doesn't return results

4. **Documentation Updates**:
   - Updated README.md to reflect the use of Deepgram
   - Updated DEPLOYMENT.md with new API key requirements
   - Created this decision document to explain the migration

## API Differences

### AssemblyAI API (Previous Implementation)

The previous implementation used AssemblyAI's SDK and required two API calls:
1. One call to get the transcript with word-level timestamps
2. A separate call to a dedicated paragraphs endpoint to get paragraph-level timestamps

```python
# Configure transcription
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(
    speech_model="nano",
    language_detection=True
)

# Get transcript
transcript = transcriber.transcribe(audio_file, config=config)

# Get paragraphs using dedicated endpoint
headers = {'authorization': ASSEMBLYAI_API_KEY}
paragraphs_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript.id}/paragraphs"
response = requests.get(paragraphs_endpoint, headers=headers)
```

### Deepgram API (New Implementation)

The new implementation uses Deepgram's SDK and requires only one API call to get both word-level timestamps and paragraphs:

```python
# Configure transcription options
options = PrerecordedOptions(
    model="nova-3",
    language="en",
    smart_format=True,
    paragraphs=True,
    punctuate=True
)

# Open the audio file
with open(audio_file, 'rb') as audio:
    # Get transcript from Deepgram
    response = await self.deepgram.listen.prerecorded.v("1").transcribe_file(audio, options)
```

## Output Format Compatibility

To maintain compatibility with downstream analysis modules, the output files maintain the same format:

1. **Word-level timestamps** (`audio_{video_id}_words.csv`):
   - Columns: `word`, `start_time`, `end_time`, `confidence`

2. **Paragraph-level timestamps** (`audio_{video_id}_paragraphs.csv`):
   - Columns: `start_time`, `end_time`, `text`

## Fallback Mechanism

If Deepgram's paragraph feature doesn't return results, a fallback mechanism was implemented to create paragraphs from sentences:

1. Group words into sentences based on punctuation
2. Group sentences into paragraphs (every 3-5 sentences)
3. Save these generated paragraphs in the same format as expected by downstream modules

## Testing

The migration was tested with sample audio files to ensure:
1. Successful transcription with Deepgram
2. Correct output file formats
3. Compatibility with downstream analysis modules

## Future Considerations

1. **Fine-tuning**: Explore Deepgram's model fine-tuning capabilities for improved accuracy on gaming content
2. **Advanced Features**: Consider using Deepgram's additional features like speaker diarization and topic detection
3. **Performance Monitoring**: Monitor transcription quality and processing time compared to AssemblyAI

## Conclusion

The migration from AssemblyAI to Deepgram maintains all the functionality required by the Klipstream Analysis project while potentially offering cost savings and performance improvements. The implementation ensures backward compatibility with existing analysis modules.
