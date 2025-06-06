/**
 * Phase 2 Testing Script
 * 
 * This script helps verify Phase 2 functionality by testing key components
 * and providing debugging information.
 */

// Test URLs for different scenarios
const TEST_URLS = {
  dashboard: 'http://localhost:3000/dash',
  videoWithNewAPI: 'http://localhost:3000/dash/video/[VIDEO_ID]', // Replace with actual video ID
  videoWithLegacyData: 'http://localhost:3000/dash/video/[LEGACY_VIDEO_ID]', // Replace with legacy video ID
  processing: 'http://localhost:3000/dash/import/[PROCESSING_VIDEO_ID]', // Replace with processing video ID
};

// Console testing functions
const testFunctions = {
  // Test useAnalysisStatus hook
  testAnalysisStatusHook: () => {
    console.log('🧪 Testing useAnalysisStatus Hook');
    console.log('1. Check if hook is polling for processing videos');
    console.log('2. Verify polling stops for completed videos');
    console.log('3. Monitor network requests in DevTools');
    
    // Check if the hook is available in React DevTools
    if (window.React) {
      console.log('✅ React is available for testing');
    } else {
      console.log('❌ React not found - check if page loaded correctly');
    }
  },

  // Test real-time updates
  testRealTimeUpdates: () => {
    console.log('🔄 Testing Real-time Updates');
    console.log('1. Look for video cards with progress bars');
    console.log('2. Watch for automatic status updates');
    console.log('3. Check progress percentage changes');
    
    // Monitor for video card updates
    const videoCards = document.querySelectorAll('[data-testid="video-card"], .video-card');
    console.log(`Found ${videoCards.length} video cards on page`);
    
    videoCards.forEach((card, index) => {
      const status = card.querySelector('.status, [data-testid="status"]');
      const progress = card.querySelector('.progress, [data-testid="progress"]');
      console.log(`Card ${index + 1}:`, {
        status: status?.textContent || 'No status found',
        progress: progress?.textContent || 'No progress found'
      });
    });
  },

  // Test new components
  testNewComponents: () => {
    console.log('🆕 Testing New Components');
    
    // Check for ResultsViewer
    const resultsViewer = document.querySelector('[data-testid="results-viewer"], .results-viewer');
    if (resultsViewer) {
      console.log('✅ ResultsViewer component found');
    } else {
      console.log('ℹ️ ResultsViewer not found (may not be on results page)');
    }
    
    // Check for HighlightsPlayer
    const highlightsPlayer = document.querySelector('[data-testid="highlights-player"], .highlights-player');
    if (highlightsPlayer) {
      console.log('✅ HighlightsPlayer component found');
    } else {
      console.log('ℹ️ HighlightsPlayer not found (may not be on results page)');
    }
    
    // Check for SentimentChart
    const sentimentChart = document.querySelector('[data-testid="sentiment-chart"], .sentiment-chart');
    if (sentimentChart) {
      console.log('✅ SentimentChart component found');
    } else {
      console.log('ℹ️ SentimentChart not found (may not be on results page)');
    }
    
    // Check for ProgressTracker
    const progressTracker = document.querySelector('[data-testid="progress-tracker"], .progress-tracker');
    if (progressTracker) {
      console.log('✅ ProgressTracker component found');
    } else {
      console.log('ℹ️ ProgressTracker not found (may not be on processing page)');
    }
  },

  // Test API integration
  testAPIIntegration: () => {
    console.log('🔌 Testing API Integration');
    
    // Monitor network requests
    console.log('Monitor these API calls in Network tab:');
    console.log('- Convex queries for video status');
    console.log('- KlipStream API status polling');
    console.log('- Analysis results fetching');
    
    // Check for error states
    const errorElements = document.querySelectorAll('.error, [data-testid="error"]');
    if (errorElements.length > 0) {
      console.log(`⚠️ Found ${errorElements.length} error elements`);
      errorElements.forEach((el, index) => {
        console.log(`Error ${index + 1}:`, el.textContent);
      });
    } else {
      console.log('✅ No error elements found');
    }
  },

  // Test responsive design
  testResponsiveDesign: () => {
    console.log('📱 Testing Responsive Design');
    
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight
    };
    
    console.log('Current viewport:', viewport);
    
    if (viewport.width < 768) {
      console.log('📱 Mobile viewport detected');
    } else if (viewport.width < 1024) {
      console.log('📟 Tablet viewport detected');
    } else {
      console.log('🖥️ Desktop viewport detected');
    }
    
    // Check for responsive elements
    const responsiveElements = document.querySelectorAll('.responsive, .grid, .flex');
    console.log(`Found ${responsiveElements.length} potentially responsive elements`);
  },

  // Run all tests
  runAllTests: () => {
    console.log('🚀 Running All Phase 2 Tests');
    console.log('================================');
    
    testFunctions.testAnalysisStatusHook();
    console.log('');
    testFunctions.testRealTimeUpdates();
    console.log('');
    testFunctions.testNewComponents();
    console.log('');
    testFunctions.testAPIIntegration();
    console.log('');
    testFunctions.testResponsiveDesign();
    
    console.log('================================');
    console.log('✅ All tests completed');
    console.log('Check console output above for results');
  }
};

// Make functions available globally for browser console
window.testPhase2 = testFunctions;

// Auto-run basic tests when script loads
console.log('🧪 Phase 2 Testing Script Loaded');
console.log('Available functions:');
console.log('- testPhase2.testAnalysisStatusHook()');
console.log('- testPhase2.testRealTimeUpdates()');
console.log('- testPhase2.testNewComponents()');
console.log('- testPhase2.testAPIIntegration()');
console.log('- testPhase2.testResponsiveDesign()');
console.log('- testPhase2.runAllTests()');
console.log('');
console.log('Run testPhase2.runAllTests() to test everything');

// Export for Node.js if needed
if (typeof module !== 'undefined' && module.exports) {
  module.exports = testFunctions;
}
