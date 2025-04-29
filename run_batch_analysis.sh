#!/bin/bash
# Script to run batch Hugging Face transcript analysis

# Check if video ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <video_id> [batch_size] [save_interval]"
    echo "Example: $0 2434635255 100 20"
    echo "  - video_id: The Twitch VOD ID"
    echo "  - batch_size: Number of entries to process (0 for all entries, default: 0)"
    echo "  - save_interval: Save results after processing this many entries (default: 50)"
    exit 1
fi

VIDEO_ID=$1
BATCH_SIZE=${2:-0}  # Default to 0 (process all entries)
SAVE_INTERVAL=${3:-50}  # Default to 50

# Install requirements
echo "Installing requirements..."
pip install -r huggingface_analysis_requirements.txt

# Check if input file exists
INPUT_FILE="outputs/audio_${VIDEO_ID}_paragraphs.csv"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p outputs
mkdir -p visualizations

# Run analysis
echo "Running batch Hugging Face transcript analysis for video $VIDEO_ID..."
echo "Saving checkpoints every $SAVE_INTERVAL entries"

# Build command
CMD="python batch_transcript_analysis.py --input_file \"$INPUT_FILE\" --output_file \"outputs/audio_${VIDEO_ID}_paragraphs_hf.csv\" --text_column \"text\" --save_interval $SAVE_INTERVAL"

# Add batch_size if specified
if [ "$BATCH_SIZE" -gt 0 ]; then
    echo "Processing $BATCH_SIZE entries (test mode)"
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

# Execute the command
eval $CMD

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "Results saved to outputs/audio_${VIDEO_ID}_paragraphs_hf.csv"
    
    # Generate visualizations
    echo "Generating visualizations..."
    python visualize_analysis.py --input_file "outputs/audio_${VIDEO_ID}_paragraphs_hf.csv" --output_dir visualizations
    
    if [ $? -eq 0 ]; then
        echo "Visualizations generated successfully in visualizations/ directory"
    else
        echo "Warning: Visualization generation failed"
    fi
    
    # Run integration with main pipeline
    echo "Would you like to integrate the results with the main pipeline? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Integrating with main pipeline..."
        python integrate_huggingface_analysis.py --video_id "$VIDEO_ID"
        
        if [ $? -eq 0 ]; then
            echo "Integration completed successfully!"
        else
            echo "Warning: Integration failed"
        fi
    fi
else
    echo "Error: Analysis failed"
    exit 1
fi
