#!/bin/bash
# Script to run optimized Hugging Face transcript analysis

# Check if video ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <video_id> [batch_size] [num_threads]"
    echo "Example: $0 2434635255 100 16"
    echo "  - video_id: The Twitch VOD ID"
    echo "  - batch_size: Number of entries to process (0 for all entries, default: 0)"
    echo "  - num_threads: Number of threads to use (default: CPU count * 2)"
    exit 1
fi

VIDEO_ID=$1
BATCH_SIZE=${2:-0}  # Default to 0 (process all entries)
NUM_THREADS=${3:-0}  # Default to 0 (auto)

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
echo "Running optimized Hugging Face transcript analysis for video $VIDEO_ID..."
if [ "$NUM_THREADS" -gt 0 ]; then
    echo "Using $NUM_THREADS threads"
fi

# Build command
CMD="python optimized_transcript_analysis.py --input_file \"$INPUT_FILE\" --output_file \"outputs/audio_${VIDEO_ID}_paragraphs_hf.csv\" --text_column \"text\""

# Add batch_size if specified
if [ "$BATCH_SIZE" -gt 0 ]; then
    echo "Processing $BATCH_SIZE entries (test mode)"
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

# Add num_threads if specified
if [ "$NUM_THREADS" -gt 0 ]; then
    CMD="$CMD --num_threads $NUM_THREADS"
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
