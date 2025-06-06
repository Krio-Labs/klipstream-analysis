// Script to check all videos in the system regardless of team
const { ConvexHttpClient } = require("convex/browser");

const convex = new ConvexHttpClient("https://laudable-horse-446.convex.cloud");

async function checkAllVideos() {
  console.log("🔍 Checking all videos in the system...\n");

  try {
    // Get videos without team filter to see all videos
    console.log("📹 Fetching all videos...");
    
    // Try to get videos with different team IDs
    const teams = [
      "j57c8hhqhqhqhqhqhqhqhqhqhqhqhqhq", // Current team
      "", // Empty team
      undefined // No team filter
    ];

    for (const team of teams) {
      console.log(`\n👥 Checking team: ${team || "undefined"}`);
      
      try {
        const params = team !== undefined ? 
          { team, paginationOpts: { numItems: 20, cursor: null } } :
          { paginationOpts: { numItems: 20, cursor: null } };
          
        const videos = await convex.query("video:list", params);
        
        console.log(`📊 Found ${videos.page?.length || 0} videos for this team`);
        
        if (videos.page && videos.page.length > 0) {
          videos.page.forEach((video, index) => {
            console.log(`  ${index + 1}. ${video.title}`);
            console.log(`     Status: ${video.status}`);
            console.log(`     Team: ${video.team}`);
            console.log(`     Created: ${new Date(video._creationTime).toLocaleString()}`);
            console.log(`     Age: ${Math.round((Date.now() - video._creationTime) / 1000 / 60)} minutes`);
            console.log(`     Current Job ID: ${video.current_job_id || "None"}`);
            console.log("");
          });
        }
      } catch (error) {
        console.log(`     Error: ${error.message}`);
      }
    }

    // Check queue status
    console.log("\n📋 Current Queue Status:");
    const queueStatus = await convex.query("queueManager:getQueueStatus", {});
    console.log("Queue Length:", queueStatus.queueLength);
    console.log("Processing Count:", queueStatus.processingCount);
    
    if (queueStatus.currentlyProcessing.length > 0) {
      console.log("\n🔄 Currently Processing:");
      queueStatus.currentlyProcessing.forEach((job, index) => {
        console.log(`  ${index + 1}. ${job.title}`);
        console.log(`     Video ID: ${job.videoId}`);
        console.log(`     Progress: ${job.progress}%`);
        console.log(`     Stage: ${job.stage}`);
        console.log(`     Started: ${job.startedAt}`);
        console.log("");
      });
    }

    if (queueStatus.queue.length > 0) {
      console.log("\n⏳ Queued Videos:");
      queueStatus.queue.forEach((job, index) => {
        console.log(`  ${index + 1}. ${job.title}`);
        console.log(`     Position: ${job.position}`);
        console.log(`     Priority: ${job.priority}`);
        console.log(`     Queued: ${job.queuedAt}`);
        console.log("");
      });
    }

  } catch (error) {
    console.error("❌ Error checking videos:", error.message);
  }
}

// Run the check
checkAllVideos().then(() => {
  console.log("✅ Video check complete!");
  process.exit(0);
}).catch(error => {
  console.error("💥 Script error:", error);
  process.exit(1);
});
