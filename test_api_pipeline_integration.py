#!/usr/bin/env python3
"""
Test Script for API-Pipeline Integration

This script tests the integration between the FastAPI application and the updated pipeline,
including transcription configuration, result mapping, and error handling.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_transcription_configuration():
    """Test transcription configuration integration"""
    logger.info("Testing transcription configuration...")
    
    try:
        from api.models import TranscriptionConfig, TranscriptionMethod
        from api.services.pipeline_wrapper import pipeline_wrapper
        
        # Test configuration creation
        config = TranscriptionConfig(
            method=TranscriptionMethod.HYBRID,
            enable_gpu=True,
            enable_fallback=True,
            cost_optimization=True
        )
        
        logger.info(f"✅ TranscriptionConfig created: {config}")
        
        # Test configuration conversion to dict
        config_dict = {
            'method': config.method.value,
            'enable_gpu': config.enable_gpu,
            'enable_fallback': config.enable_fallback,
            'cost_optimization': config.cost_optimization
        }
        
        logger.info(f"✅ Configuration dict: {config_dict}")
        
        # Test environment variable application
        original_env = pipeline_wrapper._apply_transcription_config(config_dict)
        logger.info(f"✅ Environment variables applied, original: {original_env}")
        
        # Test environment restoration
        pipeline_wrapper._restore_environment(original_env)
        logger.info("✅ Environment variables restored")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Transcription configuration test failed: {str(e)}")
        return False

async def test_result_mapping():
    """Test pipeline result mapping to API format"""
    logger.info("Testing result mapping...")
    
    try:
        from api.services.job_manager import JobManager
        from api.models import TranscriptionConfig, TranscriptionMethod
        from api.services.job_manager import AnalysisJob
        from api.models import ProcessingStage
        
        # Create test job
        job = AnalysisJob(
            id="test-job-123",
            video_id="2434635255",
            video_url="https://www.twitch.tv/videos/2434635255",
            status=ProcessingStage.COMPLETED,
            progress_percentage=100.0,
            estimated_completion_seconds=0,
            created_at=datetime.utcnow(),
            transcription_config=TranscriptionConfig(
                method=TranscriptionMethod.HYBRID,
                enable_gpu=True,
                cost_optimization=True
            )
        )
        
        # Create mock pipeline result
        mock_pipeline_result = {
            "status": "completed",
            "video_id": "2434635255",
            "files": {
                "video_file": "/tmp/video_2434635255.mp4",
                "audio_file": "/tmp/audio_2434635255.wav",
                "waveform_file": "/tmp/waveform_2434635255.json"
            },
            "uploaded_files": [
                {
                    "file_path": "video_2434635255.mp4",
                    "gcs_uri": "gs://klipstream-vods-raw/2434635255/video/video_2434635255.mp4"
                },
                {
                    "file_path": "audio_2434635255_segments.csv",
                    "gcs_uri": "gs://klipstream-transcripts/2434635255/audio_2434635255_segments.csv"
                }
            ],
            "video_duration": 3600.0,
            "transcript_word_count": 5000,
            "highlights_count": 15,
            "sentiment_score": 0.75,
            "total_duration": 450.0,
            "transcription_metadata": {
                "method_used": "hybrid",
                "cost_estimate": 0.85,
                "gpu_used": True
            }
        }
        
        # Test result mapping
        job_manager = JobManager()
        analysis_results = job_manager._map_pipeline_results_to_api_format(mock_pipeline_result, job)
        
        logger.info(f"✅ Result mapping successful")
        logger.info(f"   Video URL: {analysis_results.video_url}")
        logger.info(f"   Transcript URL: {analysis_results.transcript_url}")
        logger.info(f"   Transcription method: {analysis_results.transcription_method_used}")
        logger.info(f"   GPU used: {analysis_results.gpu_used}")
        logger.info(f"   Cost estimate: {analysis_results.transcription_cost_estimate}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Result mapping test failed: {str(e)}")
        return False

async def test_error_handling():
    """Test enhanced error handling for transcription errors"""
    logger.info("Testing enhanced error handling...")
    
    try:
        from api.services.error_handler import error_classifier
        
        # Test GPU memory error
        gpu_error = Exception("CUDA out of memory: tried to allocate 2.5GB")
        transcription_context = {
            "method_used": "parakeet",
            "gpu_available": True,
            "fallback_attempted": False
        }
        
        error_info = error_classifier.classify_transcription_error(gpu_error, transcription_context)
        logger.info(f"✅ GPU error classified: {error_info.error_code}")
        logger.info(f"   Error type: {error_info.error_type}")
        logger.info(f"   Retryable: {error_info.is_retryable}")
        logger.info(f"   Suggested action: {error_info.suggested_action}")
        
        # Test model loading error
        model_error = Exception("Failed to load Parakeet model from nvidia/parakeet-tdt-0.6b-v2")
        error_info2 = error_classifier.classify_transcription_error(model_error, transcription_context)
        logger.info(f"✅ Model error classified: {error_info2.error_code}")
        
        # Test general error classification
        network_error = Exception("Connection timeout while downloading video")
        error_info3 = error_classifier.classify_error(network_error, {"stage": "downloading"})
        logger.info(f"✅ Network error classified: {error_info3.error_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {str(e)}")
        return False

async def test_api_models():
    """Test new API models and validation"""
    logger.info("Testing API models...")
    
    try:
        from api.models import (
            AnalysisRequest, 
            TranscriptionConfig, 
            TranscriptionMethod,
            AnalysisResults,
            ProcessingStage
        )
        
        # Test AnalysisRequest with transcription config
        request = AnalysisRequest(
            url="https://www.twitch.tv/videos/2434635255",
            transcription_config=TranscriptionConfig(
                method=TranscriptionMethod.AUTO,
                enable_gpu=True,
                cost_optimization=True
            )
        )
        
        logger.info(f"✅ AnalysisRequest created with transcription config")
        logger.info(f"   URL: {request.url}")
        logger.info(f"   Method: {request.transcription_config.method}")
        
        # Test AnalysisResults with new fields
        results = AnalysisResults(
            video_url="gs://bucket/video.mp4",
            transcript_url="gs://bucket/transcript.csv",
            transcriptWords_url="gs://bucket/words.csv",
            transcription_method_used="hybrid",
            transcription_cost_estimate=0.85,
            gpu_used=True
        )
        
        logger.info(f"✅ AnalysisResults created with new fields")
        logger.info(f"   Transcription method: {results.transcription_method_used}")
        logger.info(f"   Cost estimate: {results.transcription_cost_estimate}")
        
        # Test ProcessingStage enum
        assert ProcessingStage.GENERATING_WAVEFORM == "Generating waveform"
        logger.info(f"✅ New processing stage available: {ProcessingStage.GENERATING_WAVEFORM}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API models test failed: {str(e)}")
        return False

async def test_stage_progression():
    """Test stage progression with new waveform generation stage"""
    logger.info("Testing stage progression...")
    
    try:
        from api.services.pipeline_wrapper import EnhancedPipelineWrapper
        from api.models import ProcessingStage
        
        wrapper = EnhancedPipelineWrapper()
        
        # Test stage descriptions
        stages = [
            ProcessingStage.QUEUED,
            ProcessingStage.DOWNLOADING,
            ProcessingStage.GENERATING_WAVEFORM,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.FETCHING_CHAT,
            ProcessingStage.ANALYZING,
            ProcessingStage.FINDING_HIGHLIGHTS,
            ProcessingStage.COMPLETED
        ]
        
        for stage in stages:
            description = wrapper.get_stage_description(stage)
            logger.info(f"   {stage.value}: {description}")
        
        logger.info(f"✅ All {len(stages)} stages have descriptions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stage progression test failed: {str(e)}")
        return False

async def main():
    """Run all integration tests"""
    logger.info("🚀 Starting API-Pipeline Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Transcription Configuration", test_transcription_configuration),
        ("Result Mapping", test_result_mapping),
        ("Error Handling", test_error_handling),
        ("API Models", test_api_models),
        ("Stage Progression", test_stage_progression)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        try:
            success = await test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test CRASHED: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! API-Pipeline integration is ready for production.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please review and fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
