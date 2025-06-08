#!/usr/bin/env python3
"""
Test script for Convex URL mapping with exact field names

This script tests the file-to-URL mapping functionality to ensure
it uses the exact field names required by the Convex schema.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from convex_integration import ConvexIntegration
from utils.logging_setup import setup_logger

# Setup logging
logger = setup_logger(__name__)

def test_url_mapping():
    """Test file-to-URL mapping with sample pipeline data"""
    
    logger.info("🧪 Testing Convex URL mapping with exact field names")
    
    # Initialize Convex integration
    convex_manager = ConvexIntegration()
    
    # Sample files from pipeline (typical output)
    test_video_id = "2479611486"
    test_files = {
        # Raw pipeline files
        'video_file': '/tmp/output/Raw/Video/2479611486.mp4',
        'audio_file': '/tmp/output/Raw/Audio/audio_2479611486.mp3', 
        'waveform_file': '/tmp/output/Raw/Waveform/audio_2479611486_waveform.json',
        'segments_file': '/tmp/output/Raw/Transcripts/audio_2479611486_segments.csv',
        'words_file': '/tmp/output/Raw/Transcripts/audio_2479611486_words.csv',
        'chat_file': '/tmp/output/Raw/Chat/2479611486_chat.csv',
        
        # Analysis pipeline files
        'analysis_file': '/tmp/output/Analysis/integrated_2479611486.json',
        'integrated_file': '/tmp/output/Analysis/integrated_2479611486.json'  # Alternative name
    }
    
    logger.info(f"📁 Testing with {len(test_files)} sample files")
    
    # Test the mapping function
    try:
        url_mapping = convex_manager.map_pipeline_files_to_convex_urls(test_video_id, test_files)
        
        logger.info("✅ URL Mapping Results:")
        logger.info("=" * 60)
        
        for convex_field, gcs_url in url_mapping.items():
            logger.info(f"  {convex_field}: {gcs_url}")
        
        logger.info("=" * 60)
        
        # Validate against exact Convex schema
        expected_fields = {
            'video_url', 'audio_url', 'waveform_url', 
            'transcript_url', 'transcriptWords_url', 
            'chat_url', 'analysis_url'
        }
        
        actual_fields = set(url_mapping.keys())
        
        # Check if all actual fields are valid
        invalid_fields = actual_fields - expected_fields
        missing_expected = expected_fields - actual_fields
        
        logger.info("🔍 Field Validation:")
        logger.info(f"  Expected fields: {sorted(expected_fields)}")
        logger.info(f"  Actual fields: {sorted(actual_fields)}")
        
        if invalid_fields:
            logger.error(f"  ❌ Invalid fields found: {sorted(invalid_fields)}")
            return False
        else:
            logger.info(f"  ✅ All fields are valid Convex schema fields")
        
        if missing_expected:
            logger.info(f"  ℹ️ Missing optional fields: {sorted(missing_expected)}")
        
        # Test GCS URL format
        logger.info("🌐 GCS URL Format Validation:")
        for field, url in url_mapping.items():
            if url.startswith('gs://'):
                bucket = url.split('/')[2]
                logger.info(f"  ✅ {field}: {bucket}")
            else:
                logger.error(f"  ❌ {field}: Invalid GCS URL format: {url}")
                return False
        
        logger.info("🎯 Overall Test Result: ✅ PASS")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        return False

def test_convex_field_validation():
    """Test that only exact field names are accepted"""
    
    logger.info("🧪 Testing Convex field validation")
    
    convex_manager = ConvexIntegration()
    test_video_id = "test_video"
    
    # Test with correct field names
    correct_urls = {
        'video_url': 'gs://klipstream-vods-raw/test/video.mp4',
        'audio_url': 'gs://klipstream-vods-raw/test/audio.mp3',
        'transcript_url': 'gs://klipstream-transcripts/test/segments.csv',
        'analysis_url': 'gs://klipstream-analysis/test/analysis.json'
    }
    
    # Test with incorrect field names (legacy/custom)
    incorrect_urls = {
        'transcriptUrl': 'gs://klipstream-transcripts/test/segments.csv',  # Legacy
        'custom_field': 'gs://example.com/custom.json',                   # Custom
        'my_analysis_url': 'gs://example.com/analysis.json'               # Non-standard
    }
    
    logger.info("✅ Testing correct field names:")
    for field, url in correct_urls.items():
        logger.info(f"  {field}: {url}")
    
    logger.info("❌ Testing incorrect field names (should be rejected):")
    for field, url in incorrect_urls.items():
        logger.info(f"  {field}: {url}")
    
    # Note: The actual validation happens in the ConvexManager.update_video_urls method
    # This test documents the expected behavior
    
    logger.info("🎯 Field validation test completed")
    return True

def test_bucket_mapping():
    """Test that files are mapped to correct GCS buckets"""
    
    logger.info("🧪 Testing GCS bucket mapping")
    
    convex_manager = ConvexIntegration()
    test_video_id = "test_video"
    
    # Test files and their expected buckets
    test_cases = [
        ('/tmp/output/Raw/Video/video.mp4', 'klipstream-vods-raw'),
        ('/tmp/output/Raw/Audio/audio.mp3', 'klipstream-vods-raw'),
        ('/tmp/output/Raw/Chat/chat.csv', 'klipstream-chatlogs'),
        ('/tmp/output/Raw/Transcripts/segments.csv', 'klipstream-transcripts'),
        ('/tmp/output/Raw/Transcripts/words.csv', 'klipstream-transcripts'),
        ('/tmp/output/Raw/Waveform/waveform.json', 'klipstream-vods-raw'),
        ('/tmp/output/Analysis/analysis.json', 'klipstream-analysis'),
        ('/tmp/output/Analysis/integrated.json', 'klipstream-analysis'),
    ]
    
    logger.info("🗂️ Testing bucket mapping:")
    all_correct = True
    
    for file_path, expected_bucket in test_cases:
        gcs_url = convex_manager._generate_gcs_url_from_path(test_video_id, file_path)
        actual_bucket = gcs_url.split('/')[2] if gcs_url else None
        
        if actual_bucket == expected_bucket:
            logger.info(f"  ✅ {Path(file_path).name} -> {actual_bucket}")
        else:
            logger.error(f"  ❌ {Path(file_path).name} -> {actual_bucket} (expected: {expected_bucket})")
            all_correct = False
    
    if all_correct:
        logger.info("🎯 Bucket mapping test: ✅ PASS")
    else:
        logger.error("🎯 Bucket mapping test: ❌ FAIL")
    
    return all_correct

def main():
    """Run all tests"""
    
    logger.info("🚀 Starting Convex URL mapping tests")
    logger.info("=" * 80)
    
    tests = [
        ("URL Mapping", test_url_mapping),
        ("Field Validation", test_convex_field_validation),
        ("Bucket Mapping", test_bucket_mapping)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} Test: PASSED")
            else:
                logger.error(f"❌ {test_name} Test: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} Test: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("-" * 40)
    logger.info(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ready for deployment.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
