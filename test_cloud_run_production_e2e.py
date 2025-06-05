#!/usr/bin/env python3
"""
Cloud Run Production End-to-End Test

This script tests the deployed Cloud Run service to validate:
1. FastAPI subprocess wrapper fix works in production
2. Status updates flow correctly without regressions
3. Complete pipeline works end-to-end
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


class CloudRunProductionTester:
    """Comprehensive production testing for Cloud Run deployment"""
    
    def __init__(self, base_url="https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"):
        self.base_url = base_url
        self.status_history = []
        self.test_results = {
            "deployment_health": False,
            "subprocess_fix": False,
            "status_consistency": False,
            "pipeline_completion": False,
            "overall_success": False
        }
        
    async def run_comprehensive_test(self, twitch_url, max_time=1800):  # 30 minutes max
        """Run comprehensive end-to-end test"""
        logger.info("🚀 CLOUD RUN PRODUCTION END-TO-END TEST")
        logger.info("=" * 80)
        logger.info(f"🌐 Cloud Run URL: {self.base_url}")
        logger.info(f"📹 Test Video: {twitch_url}")
        logger.info(f"⏱️ Max Time: {max_time/60:.1f} minutes")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                # Test 1: Deployment Health
                await self.test_deployment_health(session)
                
                # Test 2: Start Analysis and Monitor
                job_id = await self.start_analysis_test(session, twitch_url)
                if not job_id:
                    return self.finalize_results(False, time.time() - start_time)
                
                # Test 3: Monitor Status Consistency and Pipeline Progress
                success = await self.monitor_pipeline_execution(session, job_id, max_time)
                
                # Test 4: Validate Results
                await self.validate_completion_results(session, job_id)
                
                total_time = time.time() - start_time
                return self.finalize_results(success, total_time)
                
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            total_time = time.time() - start_time
            return self.finalize_results(False, total_time)
    
    async def test_deployment_health(self, session):
        """Test 1: Validate deployment health"""
        logger.info("\n🔍 TEST 1: DEPLOYMENT HEALTH")
        logger.info("-" * 40)
        
        try:
            # Health check
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Health Check: {data.get('status', 'unknown')}")
                    logger.info(f"📊 Version: {data.get('version', 'unknown')}")
                    self.test_results["deployment_health"] = True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
            
            # API documentation accessible
            async with session.get(f"{self.base_url}/docs") as response:
                if response.status == 200:
                    logger.info("✅ API Documentation accessible")
                else:
                    logger.warning(f"⚠️ API docs not accessible: {response.status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Health test failed: {e}")
            return False
    
    async def start_analysis_test(self, session, twitch_url):
        """Test 2: Start analysis and validate initial response"""
        logger.info("\n🚀 TEST 2: ANALYSIS INITIATION")
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
                    
                    logger.info(f"✅ Analysis started successfully")
                    logger.info(f"📋 Job ID: {job_id}")
                    logger.info(f"📊 Initial Status: {data.get('status')}")
                    logger.info(f"🎯 Initial Progress: {data.get('progress', {}).get('percentage', 0):.1f}%")
                    
                    # Record initial status
                    self.record_status_check(data.get("status"), data.get("progress", {}).get("percentage", 0))
                    
                    return job_id
                else:
                    logger.error(f"❌ Failed to start analysis: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error details: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Analysis start failed: {e}")
            return None
    
    async def monitor_pipeline_execution(self, session, job_id, max_time):
        """Test 3: Monitor pipeline execution for subprocess fix and status consistency"""
        logger.info("\n📊 TEST 3: PIPELINE EXECUTION MONITORING")
        logger.info("-" * 40)
        
        start_time = time.time()
        check_count = 0
        subprocess_fix_validated = False
        status_regressions = 0
        last_percentage = 0
        
        # Key milestones to track
        milestones = {
            "downloading_started": False,
            "transcription_started": False,
            "analysis_started": False,
            "completion": False
        }
        
        while time.time() - start_time < max_time:
            check_count += 1
            
            try:
                async with session.get(f"{self.base_url}/api/v1/analysis/{job_id}/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")
                        progress = data.get("progress", {})
                        percentage = progress.get("percentage", 0)
                        current_stage = progress.get("current_stage", "Unknown")
                        message = progress.get("message", "")
                        
                        # Record status
                        regression = self.record_status_check(status, percentage, current_stage)
                        if regression:
                            status_regressions += 1
                        
                        # Check for subprocess fix validation
                        if not subprocess_fix_validated and "download" in status.lower():
                            subprocess_fix_validated = True
                            self.test_results["subprocess_fix"] = True
                            logger.info("✅ SUBPROCESS FIX: Video download started successfully!")
                        
                        # Track milestones
                        self.track_milestones(milestones, status, current_stage)
                        
                        # Log progress
                        elapsed = time.time() - start_time
                        logger.info(f"📈 Check {check_count} ({elapsed:.1f}s): {status} | {percentage:.1f}% | {current_stage}")
                        
                        # Check for completion
                        if status in ["completed", "success"]:
                            logger.info("🎉 Pipeline completed successfully!")
                            milestones["completion"] = True
                            self.test_results["pipeline_completion"] = True
                            break
                            
                        elif status in ["failed", "error"]:
                            logger.error(f"💥 Pipeline failed: {status}")
                            logger.error(f"Error message: {message}")
                            return False
                        
                        # Check for percentage regression
                        if percentage < last_percentage and status != "completed":
                            logger.warning(f"⚠️ Percentage regression: {last_percentage:.1f}% → {percentage:.1f}%")
                        
                        last_percentage = percentage
                        
                        # Adaptive sleep based on stage
                        if "download" in status.lower():
                            await asyncio.sleep(5)  # Check more frequently during download
                        else:
                            await asyncio.sleep(10)  # Less frequent for other stages
                        
                    else:
                        logger.warning(f"⚠️ Status check failed: {response.status}")
                        await asyncio.sleep(5)
                        
            except Exception as e:
                logger.error(f"❌ Error monitoring status: {e}")
                await asyncio.sleep(5)
        
        # Evaluate status consistency
        if status_regressions == 0:
            self.test_results["status_consistency"] = True
            logger.info("✅ STATUS CONSISTENCY: No regressions detected!")
        else:
            logger.warning(f"⚠️ STATUS CONSISTENCY: {status_regressions} regressions detected")
        
        # Log milestone summary
        self.log_milestone_summary(milestones)
        
        return milestones["completion"] and status_regressions == 0
    
    def track_milestones(self, milestones, status, stage):
        """Track key pipeline milestones"""
        if not milestones["downloading_started"] and "download" in status.lower():
            milestones["downloading_started"] = True
            logger.info("🎯 MILESTONE: Video downloading started!")
            
        if not milestones["transcription_started"] and "transcrib" in status.lower():
            milestones["transcription_started"] = True
            logger.info("🎯 MILESTONE: Audio transcription started!")
            
        if not milestones["analysis_started"] and "analyz" in status.lower():
            milestones["analysis_started"] = True
            logger.info("🎯 MILESTONE: Content analysis started!")
    
    def record_status_check(self, status, percentage, stage=None):
        """Record status and detect regressions"""
        timestamp = datetime.now()
        
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
        
        status_record["regression"] = regression_detected
        self.status_history.append(status_record)
        
        return regression_detected
    
    def log_milestone_summary(self, milestones):
        """Log summary of achieved milestones"""
        logger.info("\n🎯 MILESTONE SUMMARY:")
        for milestone, achieved in milestones.items():
            status_icon = "✅" if achieved else "❌"
            logger.info(f"  {status_icon} {milestone.replace('_', ' ').title()}: {'Achieved' if achieved else 'Not Achieved'}")
    
    async def validate_completion_results(self, session, job_id):
        """Test 4: Validate completion results if pipeline succeeded"""
        if not self.test_results["pipeline_completion"]:
            logger.info("\n⏭️ TEST 4: SKIPPED (Pipeline did not complete)")
            return
            
        logger.info("\n🔍 TEST 4: COMPLETION VALIDATION")
        logger.info("-" * 40)
        
        try:
            # Get final status
            async with session.get(f"{self.base_url}/api/v1/analysis/{job_id}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Final Status: {data.get('status')}")
                    logger.info(f"📊 Final Progress: {data.get('progress', {}).get('percentage', 0):.1f}%")
                    
                    # Check if results are available
                    results = data.get("results")
                    if results:
                        logger.info("✅ Results available:")
                        for key, value in results.items():
                            if isinstance(value, str) and value.startswith("http"):
                                logger.info(f"  📄 {key}: {value}")
                            else:
                                logger.info(f"  📊 {key}: {value}")
                    else:
                        logger.warning("⚠️ No results data available")
                        
        except Exception as e:
            logger.error(f"❌ Completion validation failed: {e}")
    
    def finalize_results(self, success, total_time):
        """Finalize and report test results"""
        logger.info("\n" + "🎯" * 40)
        logger.info("COMPREHENSIVE TEST RESULTS")
        logger.info("🎯" * 40)
        
        logger.info(f"⏱️ Total Test Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"📊 Total Status Checks: {len(self.status_history)}")
        
        # Individual test results
        logger.info("\n📋 INDIVIDUAL TEST RESULTS:")
        for test_name, result in self.test_results.items():
            if test_name == "overall_success":
                continue
            status_icon = "✅" if result else "❌"
            logger.info(f"  {status_icon} {test_name.replace('_', ' ').title()}: {'PASS' if result else 'FAIL'}")
        
        # Status regression analysis
        regressions = [r for r in self.status_history if r.get("regression", False)]
        logger.info(f"\n🚨 Status Regressions: {len(regressions)}")
        
        # Overall assessment
        self.test_results["overall_success"] = all([
            self.test_results["deployment_health"],
            self.test_results["subprocess_fix"],
            self.test_results["status_consistency"],
            self.test_results["pipeline_completion"]
        ])
        
        logger.info("\n🎯 OVERALL ASSESSMENT:")
        if self.test_results["overall_success"]:
            logger.info("✅ ALL TESTS PASSED - Production deployment is fully functional!")
            logger.info("✅ FastAPI subprocess fix working in Cloud Run")
            logger.info("✅ Status updates flowing correctly")
            logger.info("✅ Complete pipeline working end-to-end")
        else:
            logger.error("❌ SOME TESTS FAILED - Issues detected in production deployment")
            
        return self.test_results


async def main():
    """Main test function"""
    # Use the same test video for consistency
    test_video = "https://www.twitch.tv/videos/2434635255"
    
    logger.info("🧪 CLOUD RUN PRODUCTION END-TO-END TEST")
    logger.info("=" * 80)
    logger.info("This test validates the complete production deployment:")
    logger.info("  ✅ Deployment health and accessibility")
    logger.info("  ✅ FastAPI subprocess wrapper fix in Cloud Run")
    logger.info("  ✅ Status update consistency and flow")
    logger.info("  ✅ Complete pipeline execution")
    logger.info("=" * 80)
    
    tester = CloudRunProductionTester()
    results = await tester.run_comprehensive_test(test_video)
    
    return results["overall_success"]


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
