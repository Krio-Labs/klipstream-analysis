# Transcript Analysis with Hugging Face Models

This set of scripts provides tools for analyzing transcript data using Hugging Face's sentiment and emotion analysis models. The scripts integrate with the existing pipeline and provide visualizations to compare different analysis methods.

## Models Used

- **Sentiment Analysis**: [llmware/slim-sentiment-tool](https://huggingface.co/llmware/slim-sentiment-tool)
- **Emotion Analysis**: [llmware/slim-emotions-tool](https://huggingface.co/llmware/slim-emotions-tool)

## Files Included

1. **test_transcript_analysis.py**: A simple test script for analyzing transcript data.
2. **huggingface_transcript_analysis.py**: The main script for analyzing transcript data and comparing with existing sentiment analysis.
3. **visualize_analysis.py**: A script for creating visualizations of the analysis results.
4. **run_huggingface_analysis.sh**: A shell script for running the analysis pipeline.
5. **huggingface_analysis_requirements.txt**: A list of required Python packages.

## Installation

Install the required packages:

```bash
pip install -r huggingface_analysis_requirements.txt
```

## Usage

### Quick Start

The easiest way to run the analysis is to use the provided shell script:

```bash
chmod +x run_huggingface_analysis.sh
./run_huggingface_analysis.sh VIDEO_ID [BATCH_SIZE]
```

Where:
- `VIDEO_ID` is the ID of the Twitch VOD (e.g., 2434635255)
- `BATCH_SIZE` (optional) is the number of entries to process for testing (default: process all entries)

### Manual Usage

1. **Run the Transcript Analysis**:

```bash
python huggingface_transcript_analysis.py --input_file outputs/audio_VIDEO_ID_paragraphs.csv --output_file outputs/audio_VIDEO_ID_paragraphs_hf.csv
```

2. **Create Visualizations**:

```bash
python visualize_analysis.py --input_file outputs/audio_VIDEO_ID_paragraphs_hf.csv --output_dir visualizations
```

## Visualizations

The visualization script creates the following charts:

1. **Sentiment Comparison**: A heatmap comparing the existing sentiment analysis with the Hugging Face sentiment analysis.
2. **Emotion Distribution**: A bar chart showing the distribution of emotions in the transcript.
3. **Sentiment Timeline**: A line chart showing sentiment over time.
4. **Emotion Timeline**: A multi-panel chart showing the occurrence of the top 5 emotions over time.

## Testing

To test the scripts with a small subset of data:

```bash
python huggingface_transcript_analysis.py --input_file outputs/audio_VIDEO_ID_paragraphs.csv --output_file outputs/audio_VIDEO_ID_paragraphs_hf.csv --batch_size 10
```

## Integration with Existing Pipeline

These scripts are designed to work alongside the existing pipeline. They take the output of the transcription step (the paragraphs CSV file) and add additional sentiment and emotion analysis using Hugging Face models.

The results can be used to:
1. Compare different sentiment analysis methods
2. Provide more detailed emotion analysis
3. Identify potential highlight moments based on emotional content

## Future Improvements

1. Add support for more Hugging Face models
2. Integrate the analysis directly into the main pipeline
3. Use the emotion analysis to improve highlight detection
4. Add more advanced visualizations and analytics
