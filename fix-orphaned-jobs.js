// Fix orphaned jobs that are confusing the queue system
const { ConvexHttpClient } = require("convex/browser");

const convex = new ConvexHttpClient("https://laudable-horse-446.convex.cloud");

async function fixOrphanedJobs() {
  console.log("🔧 Fixing orphaned jobs in the database...\n");

  try {
    // Step 1: Get all jobs to analyze the current state
    console.log("📊 Step 1: Analyzing current job states...");
    
    // We'll use a direct query to get all jobs
    const allJobs = await convex.query("queueManager:getQueueStatus", {});
    
    console.log(`Found ${allJobs.queueLength} queued jobs`);
    console.log(`Found ${allJobs.processingCount} processing jobs`);
    
    if (allJobs.queue.length > 0) {
      console.log("\n📋 Queued jobs:");
      allJobs.queue.forEach((job, index) => {
        console.log(`  ${index + 1}. ${job.title || 'Unknown'}`);
        console.log(`     Job ID: ${job.jobId}`);
        console.log(`     Position: ${job.position}`);
        console.log(`     Queued at: ${job.queuedAt}`);
      });
    }
    
    if (allJobs.currentlyProcessing.length > 0) {
      console.log("\n🔄 Processing jobs:");
      allJobs.currentlyProcessing.forEach((job, index) => {
        console.log(`  ${index + 1}. ${job.title || 'Unknown'}`);
        console.log(`     Job ID: ${job.jobId}`);
        console.log(`     Video ID: ${job.videoId}`);
        console.log(`     Stage: ${job.stage}`);
        console.log(`     Started: ${job.startedAt}`);
      });
    }

    // Step 2: Clear stuck jobs (this should handle jobs that are stuck in processing)
    console.log("\n🧹 Step 2: Clearing stuck jobs...");
    const clearResult = await convex.mutation("queueManager:clearStuckJobs", {});
    console.log(`✅ Cleared ${clearResult.clearedCount} stuck jobs`);

    // Step 3: Check for stuck videos and fix them
    console.log("\n📹 Step 3: Checking and fixing stuck videos...");
    const stuckVideoResult = await convex.mutation("jobs:checkStuckVideos", {});
    console.log(`✅ Fixed ${stuckVideoResult.stuckVideosFound} stuck videos`);

    if (stuckVideoResult.updatedVideos.length > 0) {
      console.log("📝 Updated videos:");
      stuckVideoResult.updatedVideos.forEach(video => {
        console.log(`  - ${video.title} (${video.id})`);
        console.log(`    Previous status: ${video.previousStatus}`);
        console.log(`    New status: failed (due to age: ${video.ageMinutes} minutes)`);
      });
    }

    // Step 4: Final status check
    console.log("\n📊 Step 4: Final status after cleanup...");
    const finalStatus = await convex.query("queueManager:getQueueStatus", {});
    console.log(`Final queue length: ${finalStatus.queueLength}`);
    console.log(`Final processing count: ${finalStatus.processingCount}`);

    // Step 5: Summary and recommendations
    console.log("\n" + "=".repeat(50));
    console.log("🎉 ORPHANED JOBS CLEANUP COMPLETE");
    console.log("=".repeat(50));
    
    console.log(`✅ Stuck jobs cleared: ${clearResult.clearedCount}`);
    console.log(`✅ Stuck videos fixed: ${stuckVideoResult.stuckVideosFound}`);
    console.log(`📊 Final queue state: ${finalStatus.queueLength} queued, ${finalStatus.processingCount} processing`);

    if (finalStatus.queueLength === 0 && finalStatus.processingCount === 0) {
      console.log("\n🎯 PERFECT! Queue is now clean and ready for new videos.");
      console.log("💡 Next steps:");
      console.log("  1. Refresh your browser to clear cache");
      console.log("  2. Submit a new video to test the queue");
      console.log("  3. Verify it processes correctly");
    } else if (finalStatus.processingCount === 1 && finalStatus.queueLength === 0) {
      console.log("\n✅ GOOD! One video is processing, queue is empty.");
      console.log("💡 This is the expected state. When this video completes,");
      console.log("   the queue will be ready for new submissions.");
    } else {
      console.log("\n⚠️  There may still be some issues. Manual intervention might be needed.");
    }

  } catch (error) {
    console.error("❌ Error fixing orphaned jobs:", error.message);
    console.error("Stack:", error.stack);
  }
}

// Run the fix
fixOrphanedJobs().then(() => {
  console.log("\n✅ Orphaned jobs fix completed!");
  process.exit(0);
}).catch(error => {
  console.error("💥 Fix error:", error);
  process.exit(1);
});
