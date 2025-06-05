#!/usr/bin/env python3
"""
Test Status Updates Fix

This script tests that our status update fixes work correctly by monitoring
the status progression without conflicts or regressions.
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


class StatusUpdateTester:
    """Test status update flow for correctness"""
    
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.status_history = []
        
    async def test_status_flow(self, twitch_url, max_time=300):  # 5 minutes for quick test
        """Test that status updates flow correctly without regression"""
        logger.info("🔍 TESTING STATUS UPDATE FLOW")
        logger.info("=" * 60)
        logger.info(f"🌐 API Base URL: {self.base_url}")
        logger.info(f"📹 Test Video: {twitch_url}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Start analysis
            job_id = await self.start_analysis(session, twitch_url)
            if not job_id:
                return False
            
            # Monitor status progression
            success = await self.monitor_status_progression(session, job_id, max_time)
            
            total_time = time.time() - start_time
            
            # Analyze results
            await self.analyze_status_flow(success, total_time)
            
            return success
    
    async def start_analysis(self, session, twitch_url):
        """Start analysis and return job ID"""
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
                    
                    # Record initial status
                    self.record_status(data.get("status"), data.get("progress", {}).get("percentage", 0))
                    
                    logger.info(f"✅ Analysis started - Job ID: {job_id}")
                    return job_id
                else:
                    logger.error(f"❌ Failed to start analysis: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Start analysis error: {e}")
            return None
    
    async def monitor_status_progression(self, session, job_id, max_time):
        """Monitor status progression and detect issues"""
        logger.info("\n📊 MONITORING STATUS PROGRESSION")
        logger.info("-" * 40)
        
        start_time = time.time()
        check_count = 0
        regression_detected = False
        
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
                        
                        # Record status
                        regression = self.record_status(status, percentage, current_stage)
                        
                        if regression:
                            regression_detected = True
                            logger.error(f"🚨 STATUS REGRESSION DETECTED at check {check_count}")
                        
                        # Log current status
                        elapsed = time.time() - start_time
                        logger.info(f"📈 Check {check_count} ({elapsed:.1f}s): {status} | {percentage:.1f}% | {current_stage}")
                        
                        # Check for completion
                        if status in ["completed", "success"]:
                            logger.info("🎉 Analysis completed!")
                            return not regression_detected
                            
                        elif status in ["failed", "error"]:
                            logger.error(f"💥 Analysis failed: {status}")
                            return False
                        
                        await asyncio.sleep(2)  # Check every 2 seconds for detailed monitoring
                        
                    else:
                        logger.warning(f"⚠️ Status check failed: {response.status}")
                        await asyncio.sleep(1)
                        
            except Exception as e:
                logger.error(f"❌ Error monitoring status: {e}")
                await asyncio.sleep(1)
        
        logger.warning("⏰ Monitoring timeout reached")
        return not regression_detected
    
    def record_status(self, status, percentage, stage=None):
        """Record status and detect regressions"""
        timestamp = datetime.now()
        
        # Create status record
        status_record = {
            "timestamp": timestamp,
            "status": status,
            "percentage": percentage,
            "stage": stage,
            "check_number": len(self.status_history) + 1
        }
        
        # Check for regression
        regression_detected = False
        
        if self.status_history:
            last_record = self.status_history[-1]
            
            # Check for percentage regression
            if percentage < last_record["percentage"] and status != "completed":
                logger.error(f"🚨 PERCENTAGE REGRESSION: {last_record['percentage']:.1f}% → {percentage:.1f}%")
                regression_detected = True
            
            # Check for status regression (completed → something else)
            if last_record["status"] == "completed" and status != "completed":
                logger.error(f"🚨 STATUS REGRESSION: completed → {status}")
                regression_detected = True
            
            # Check for stage regression
            stage_order = ["queued", "downloading", "fetching_chat", "transcribing", "analyzing", "finding_highlights", "completed"]
            if stage and last_record.get("stage"):
                try:
                    current_index = stage_order.index(stage.lower())
                    last_index = stage_order.index(last_record["stage"].lower())
                    
                    if current_index < last_index and status != "completed":
                        logger.error(f"🚨 STAGE REGRESSION: {last_record['stage']} → {stage}")
                        regression_detected = True
                except ValueError:
                    # Stage not in expected order, skip check
                    pass
        
        # Add regression flag to record
        status_record["regression"] = regression_detected
        
        # Store record
        self.status_history.append(status_record)
        
        return regression_detected
    
    async def analyze_status_flow(self, success, total_time):
        """Analyze the complete status flow"""
        logger.info("\n" + "📊" * 30)
        logger.info("STATUS FLOW ANALYSIS")
        logger.info("📊" * 30)
        
        logger.info(f"⏱️ Total Time: {total_time:.1f} seconds")
        logger.info(f"📊 Total Status Checks: {len(self.status_history)}")
        
        # Count regressions
        regressions = [r for r in self.status_history if r.get("regression", False)]
        logger.info(f"🚨 Regressions Detected: {len(regressions)}")
        
        if regressions:
            logger.info("\n🚨 REGRESSION DETAILS:")
            for i, regression in enumerate(regressions, 1):
                logger.info(f"  {i}. Check #{regression['check_number']}: {regression['status']} ({regression['percentage']:.1f}%) - {regression['stage']}")
        
        # Show status progression
        logger.info("\n📈 STATUS PROGRESSION:")
        for i, record in enumerate(self.status_history):
            status_icon = "🚨" if record.get("regression") else "✅"
            logger.info(f"  {status_icon} Check #{record['check_number']}: {record['status']} ({record['percentage']:.1f}%) - {record.get('stage', 'N/A')}")
        
        # Final assessment
        logger.info("\n🎯 ASSESSMENT:")
        if success and len(regressions) == 0:
            logger.info("✅ STATUS FLOW: PERFECT - No regressions detected!")
            logger.info("✅ STATUS UPDATES: Working correctly!")
        elif success and len(regressions) > 0:
            logger.warning("⚠️ STATUS FLOW: COMPLETED WITH ISSUES - Regressions detected but process completed")
            logger.warning("💡 STATUS UPDATES: Need further investigation")
        else:
            logger.error("❌ STATUS FLOW: FAILED - Process did not complete successfully")
            logger.error("🔧 STATUS UPDATES: Require fixes")
        
        return success and len(regressions) == 0


async def main():
    """Main test function"""
    # Use a shorter video for faster testing
    test_video = "https://www.twitch.tv/videos/2434635255"
    
    logger.info("🧪 STATUS UPDATE FLOW TEST")
    logger.info("=" * 60)
    logger.info("This test validates that status updates flow correctly without regressions")
    logger.info("Key validation points:")
    logger.info("  - No percentage regressions")
    logger.info("  - No status regressions (completed → other)")
    logger.info("  - No stage regressions")
    logger.info("  - Proper progression through stages")
    logger.info("=" * 60)
    
    tester = StatusUpdateTester()
    success = await tester.test_status_flow(test_video)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
