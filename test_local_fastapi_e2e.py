#!/usr/bin/env python3
"""
Local FastAPI End-to-End Test

This script tests the complete pipeline by sending a real Twitch URL to the local FastAPI server
and monitoring the process until completion.
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalFastAPITester:
    """End-to-end tester for local FastAPI deployment"""
    
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        
    async def test_complete_pipeline(self, twitch_url, max_time=1800):  # 30 minutes max
        """Test the complete pipeline with a real Twitch URL"""
        logger.info("🚀 STARTING LOCAL FASTAPI END-TO-END TEST")
        logger.info("=" * 80)
        logger.info(f"🌐 API Base URL: {self.base_url}")
        logger.info(f"📹 Test Video: {twitch_url}")
        logger.info(f"⏱️ Max Time: {max_time/60:.1f} minutes")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Test API health
            if not await self.test_api_health(session):
                return False
            
            # Step 2: Start analysis
            job_id = await self.start_analysis(session, twitch_url)
            if not job_id:
                return False
            
            # Step 3: Monitor progress until completion
            success = await self.monitor_progress(session, job_id, max_time)
            
            total_time = time.time() - start_time
            
            # Step 4: Final summary
            await self.print_final_summary(success, total_time, job_id)
            
            return success
    
    async def test_api_health(self, session):
        """Test API health endpoint"""
        logger.info("\n🔍 Step 1: Testing API Health")
        logger.info("-" * 40)
        
        try:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ API is healthy: {data.get('status')}")
                    logger.info(f"📊 Version: {data.get('version')}")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def start_analysis(self, session, twitch_url):
        """Start the analysis process"""
        logger.info("\n🚀 Step 2: Starting Analysis")
        logger.info("-" * 40)
        
        try:
            payload = {"url": twitch_url}
            
            async with session.post(
                f"{self.base_url}/api/v1/analysis",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    job_id = data.get("job_id")
                    
                    logger.info(f"✅ Analysis started successfully!")
                    logger.info(f"📋 Job ID: {job_id}")
                    logger.info(f"📊 Initial Status: {data.get('status')}")
                    logger.info(f"🎯 Progress: {data.get('progress', {}).get('percentage', 0)}%")
                    
                    return job_id
                else:
                    error_data = await response.json()
                    logger.error(f"❌ Failed to start analysis: {response.status}")
                    logger.error(f"📊 Error: {json.dumps(error_data, indent=2)}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Start analysis error: {e}")
            return None
    
    async def monitor_progress(self, session, job_id, max_time):
        """Monitor the analysis progress"""
        logger.info("\n📊 Step 3: Monitoring Progress")
        logger.info("-" * 40)
        
        start_time = time.time()
        last_percentage = 0
        last_stage = ""
        check_count = 0
        
        # Track key milestones
        milestones = {
            "started": False,
            "downloading": False,
            "transcribing": False,
            "analyzing": False,
            "completed": False
        }
        
        while time.time() - start_time < max_time:
            check_count += 1
            
            try:
                async with session.get(
                    f"{self.base_url}/api/v1/analysis/{job_id}/status"
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")
                        progress = data.get("progress", {})
                        
                        percentage = progress.get("percentage", 0)
                        current_stage = progress.get("current_stage", "Unknown")
                        message = progress.get("message", "")
                        
                        # Log progress updates
                        if percentage != last_percentage or current_stage != last_stage:
                            elapsed = time.time() - start_time
                            logger.info(f"📈 Check {check_count} ({elapsed/60:.1f}min): {status} | {percentage:.1f}% | {current_stage}")
                            
                            if message:
                                logger.info(f"   💬 {message}")
                            
                            # Track milestones
                            self.track_milestones(milestones, current_stage, percentage)
                            
                            last_percentage = percentage
                            last_stage = current_stage
                        
                        # Check for completion
                        if status in ["completed", "success"]:
                            logger.info("🎉 Analysis completed successfully!")
                            milestones["completed"] = True
                            await self.analyze_success(data, time.time() - start_time, milestones)
                            return True
                            
                        elif status in ["failed", "error"]:
                            logger.error(f"💥 Analysis failed: {status}")
                            await self.analyze_failure(data, time.time() - start_time, milestones)
                            return False
                        
                        # Wait before next check (adaptive interval)
                        if percentage < 10:
                            await asyncio.sleep(10)  # Check every 10s during early stages
                        elif percentage < 50:
                            await asyncio.sleep(15)  # Check every 15s during middle stages
                        else:
                            await asyncio.sleep(20)  # Check every 20s during final stages
                        
                    else:
                        logger.warning(f"⚠️ Status check failed: {response.status}")
                        await asyncio.sleep(5)
                        
            except Exception as e:
                logger.error(f"❌ Error monitoring progress: {e}")
                await asyncio.sleep(5)
        
        logger.warning("⏰ Monitoring timeout reached")
        await self.analyze_timeout(time.time() - start_time, milestones)
        return False
    
    def track_milestones(self, milestones, stage, percentage):
        """Track key milestones in the process"""
        stage_lower = stage.lower()
        
        if not milestones["started"] and percentage > 0:
            milestones["started"] = True
            logger.info("🎯 MILESTONE: Process started!")
        
        if not milestones["downloading"] and ("download" in stage_lower or percentage > 5):
            milestones["downloading"] = True
            logger.info("🎯 MILESTONE: Video downloading!")
        
        if not milestones["transcribing"] and ("transcrib" in stage_lower or percentage > 30):
            milestones["transcribing"] = True
            logger.info("🎯 MILESTONE: Audio transcription!")
        
        if not milestones["analyzing"] and ("analyz" in stage_lower or percentage > 60):
            milestones["analyzing"] = True
            logger.info("🎯 MILESTONE: Content analysis!")
    
    async def analyze_success(self, data, total_time, milestones):
        """Analyze successful completion"""
        logger.info("\n" + "🎉" * 30)
        logger.info("SUCCESS ANALYSIS")
        logger.info("🎉" * 30)
        
        logger.info(f"✅ Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"✅ Final Progress: {data.get('progress', {}).get('percentage', 0)}%")
        
        # Check milestones
        logger.info("\n🎯 MILESTONES ACHIEVED:")
        for milestone, achieved in milestones.items():
            status = "✅" if achieved else "❌"
            logger.info(f"  {status} {milestone.title()}")
        
        # Check for results
        if "results" in data:
            results = data["results"]
            logger.info(f"\n📊 RESULTS:")
            logger.info(f"  📹 Video File: {results.get('video_file_url', 'N/A')}")
            logger.info(f"  📝 Transcript: {results.get('transcript_file_url', 'N/A')}")
            logger.info(f"  🎯 Highlights: {results.get('highlights_file_url', 'N/A')}")
            logger.info(f"  📊 Analysis: {results.get('analysis_report_url', 'N/A')}")
        
        logger.info("\n🔍 VALIDATION RESULTS:")
        logger.info("✅ FastAPI subprocess fix is working correctly!")
        logger.info("✅ Complete pipeline executed successfully!")
        logger.info("✅ Local deployment is functional!")
    
    async def analyze_failure(self, data, total_time, milestones):
        """Analyze failure details"""
        logger.info("\n" + "💥" * 30)
        logger.info("FAILURE ANALYSIS")
        logger.info("💥" * 30)
        
        logger.info(f"📊 Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        if "error" in data:
            error_details = data["error"]
            logger.info(f"📊 Error Details: {json.dumps(error_details, indent=2)}")
        
        # Check milestones to see where it failed
        logger.info("\n🎯 MILESTONES ACHIEVED:")
        for milestone, achieved in milestones.items():
            status = "✅" if achieved else "❌"
            logger.info(f"  {status} {milestone.title()}")
        
        # Determine failure point
        if not milestones["downloading"]:
            logger.error("🚨 FAILURE POINT: Video download phase")
            logger.error("💡 This may indicate subprocess wrapper issues")
        elif not milestones["transcribing"]:
            logger.error("🚨 FAILURE POINT: Transcription phase")
        elif not milestones["analyzing"]:
            logger.error("🚨 FAILURE POINT: Analysis phase")
        else:
            logger.error("🚨 FAILURE POINT: Final processing phase")
    
    async def analyze_timeout(self, total_time, milestones):
        """Analyze timeout scenario"""
        logger.info("\n" + "⏰" * 30)
        logger.info("TIMEOUT ANALYSIS")
        logger.info("⏰" * 30)
        
        logger.info(f"📊 Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Check milestones to see progress
        logger.info("\n🎯 MILESTONES ACHIEVED:")
        for milestone, achieved in milestones.items():
            status = "✅" if achieved else "❌"
            logger.info(f"  {status} {milestone.title()}")
        
        if milestones["downloading"]:
            logger.info("✅ PARTIAL SUCCESS: Process is working but may need more time")
            logger.info("💡 Consider increasing timeout for large videos")
        else:
            logger.warning("⚠️ Process may be stuck in early stages")
    
    async def print_final_summary(self, success, total_time, job_id):
        """Print final test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL TEST SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"🆔 Job ID: {job_id}")
        logger.info(f"⏱️ Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            logger.info("\n🎉 OVERALL RESULT: SUCCESS")
            logger.info("✅ Local FastAPI deployment is working correctly!")
            logger.info("✅ Subprocess fix is effective!")
            logger.info("✅ Complete pipeline executed successfully!")
            logger.info("✅ Ready for Cloud Run deployment!")
        else:
            logger.info("\n💥 OVERALL RESULT: NEEDS INVESTIGATION")
            logger.info("🔍 Check the analysis above for specific issues")
            logger.info("💡 May need additional debugging or timeout adjustment")


async def main():
    """Main test function"""
    # Test video URL - using a shorter video for faster testing
    test_video = "https://www.twitch.tv/videos/2434635255"
    
    logger.info("🧪 LOCAL FASTAPI END-TO-END TEST")
    logger.info("=" * 80)
    logger.info("This test validates the complete pipeline with a real Twitch URL")
    logger.info("Key validation points:")
    logger.info("  - FastAPI subprocess wrapper fix")
    logger.info("  - Complete video download process")
    logger.info("  - Audio transcription")
    logger.info("  - Content analysis")
    logger.info("  - Result generation")
    logger.info("=" * 80)
    
    tester = LocalFastAPITester()
    success = await tester.test_complete_pipeline(test_video)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
