// Test script to check Convex queue status and manually trigger processing
const { ConvexHttpClient } = require("convex/browser");

const client = new ConvexHttpClient("https://laudable-horse-446.convex.cloud");

async function checkConvexQueue() {
  try {
    console.log("🔍 Checking Convex queue status...");
    
    // Get queue status
    const queueStatus = await client.query("queueManager:getQueueStatus", {});
    console.log("📊 Queue Status:");
    console.log(`   Queue length: ${queueStatus.queueLength}`);
    console.log(`   Processing count: ${queueStatus.processingCount}`);
    console.log(`   Currently processing: ${queueStatus.currentlyProcessing.length}`);
    
    if (queueStatus.queue.length > 0) {
      console.log("\n📋 Jobs in queue:");
      queueStatus.queue.forEach((job, index) => {
        console.log(`   ${index + 1}. Video ${job.videoId} (Position: ${job.position})`);
      });
    }
    
    if (queueStatus.currentlyProcessing.length > 0) {
      console.log("\n⚡ Currently processing:");
      queueStatus.currentlyProcessing.forEach((job, index) => {
        console.log(`   ${index + 1}. Job ${job.jobId} - Video ${job.videoId} (${job.progress}%)`);
      });
    }
    
    // Get next job in queue
    const nextJob = await client.query("queueManager:getNextJobInQueue", {});
    if (nextJob) {
      console.log("\n🎯 Next job in queue:");
      console.log(`   Job ID: ${nextJob.jobId}`);
      console.log(`   Video ID: ${nextJob.videoId}`);
      console.log(`   Twitch ID: ${nextJob.twitchId}`);
      console.log(`   Queue Position: ${nextJob.queuePosition}`);
      
      // Try to manually trigger processing
      console.log("\n🚀 Attempting to trigger processing...");
      try {
        const processResult = await client.mutation("processor:processQueuedVideo", {});
        console.log("✅ Process trigger result:", processResult);
      } catch (error) {
        console.log("❌ Failed to trigger processing:", error.message);
      }
    } else {
      console.log("\n✅ No jobs in queue");
    }
    
  } catch (error) {
    console.error("❌ Error checking queue:", error);
  }
}

async function addTestJobToConvexQueue() {
  try {
    console.log("\n🧪 Adding test job to Convex queue...");
    
    // First, let's see if we can find the video
    const videoId = "2479611486"; // The Twitch ID from our test
    const teamId = "js7bj9zgdkyj9ykvr4m6jarxkh7ep9fa"; // From the memories
    const jobId = `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    console.log(`   Video ID: ${videoId}`);
    console.log(`   Team ID: ${teamId}`);
    console.log(`   Job ID: ${jobId}`);
    
    // Try to add to queue
    const result = await client.mutation("queueManager:addVideoToQueue", {
      videoId: videoId,
      teamId: teamId,
      jobId: jobId,
      priority: 0
    });
    
    console.log("✅ Added to Convex queue:", result);
    
    // Check queue status again
    await checkConvexQueue();
    
  } catch (error) {
    console.error("❌ Error adding to Convex queue:", error);
  }
}

async function main() {
  console.log("🚀 Testing Convex Queue System");
  console.log("=" * 50);
  
  await checkConvexQueue();
  
  // If no jobs in queue, try adding one
  console.log("\n" + "=" * 50);
  await addTestJobToConvexQueue();
}

main().catch(console.error);
