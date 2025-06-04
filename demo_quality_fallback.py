#!/usr/bin/env python3
"""
Quality Fallback Demo Script

This script demonstrates the progressive quality fallback system
by simulating different memory and timeout scenarios.
"""

import asyncio
import sys
import logging
import psutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from raw_pipeline.enhanced_downloader import EnhancedTwitchDownloader
from utils.logging_setup import setup_logger

# Set up logging
logger = setup_logger("demo_quality_fallback", "demo_quality_fallback.log")

class QualityFallbackDemo:
    """Demo class for quality fallback functionality"""
    
    def __init__(self):
        self.downloader = EnhancedTwitchDownloader()
        
    async def run_demo(self):
        """Run the quality fallback demonstration"""
        logger.info("🎬 Quality Fallback System Demonstration")
        logger.info("=" * 60)
        
        # Show system information
        await self.show_system_info()
        
        # Demonstrate quality recommendations
        await self.demo_quality_recommendations()
        
        # Demonstrate thread optimization
        await self.demo_thread_optimization()
        
        # Demonstrate progressive fallback configuration
        await self.demo_fallback_configuration()
        
        # Show memory-based recommendations
        await self.demo_memory_scenarios()
        
        logger.info("\n🎉 Demo completed! The quality fallback system is ready to handle memory and timeout issues.")
        
    async def show_system_info(self):
        """Show current system information"""
        logger.info("\n📊 Current System Information:")
        logger.info("-" * 40)
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"Total Memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available Memory: {memory.available / (1024**2):.1f} MB")
        logger.info(f"Memory Usage: {memory.percent:.1f}%")
        
        # CPU info
        logger.info(f"CPU Cores: {psutil.cpu_count()}")
        logger.info(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
        
        # Recommended quality for current system
        available_mb = memory.available / (1024**2)
        recommended = self.downloader.get_quality_recommendation(int(available_mb))
        logger.info(f"Recommended Quality: {recommended}")
        
    async def demo_quality_recommendations(self):
        """Demonstrate quality recommendation algorithm"""
        logger.info("\n🎯 Quality Recommendation Examples:")
        logger.info("-" * 40)
        
        scenarios = [
            (8192, None, "High-end system"),
            (4096, None, "Mid-range system"),
            (2048, None, "Low-memory system"),
            (1024, None, "Constrained environment"),
            (8192, 300, "High-end system, 5-hour video"),
            (4096, 180, "Mid-range system, 3-hour video"),
            (2048, 120, "Low-memory system, 2-hour video"),
        ]
        
        for memory_mb, duration_min, description in scenarios:
            recommended = self.downloader.get_quality_recommendation(memory_mb, duration_min)
            duration_str = f", {duration_min}min video" if duration_min else ""
            logger.info(f"{description} ({memory_mb}MB{duration_str}): {recommended}")
            
    async def demo_thread_optimization(self):
        """Demonstrate thread optimization"""
        logger.info("\n⚙️ Thread Optimization Examples:")
        logger.info("-" * 40)
        
        scenarios = [
            ("720p", 8192, "High quality, high memory"),
            ("720p", 1024, "High quality, low memory"),
            ("480p", 4096, "Medium quality, medium memory"),
            ("360p", 2048, "Low quality, low memory"),
            ("worst", 1024, "Worst quality, constrained memory"),
        ]
        
        for quality, memory_mb, description in scenarios:
            threads = self.downloader._get_optimal_threads(quality, memory_mb)
            logger.info(f"{description}: {quality} with {memory_mb}MB → {threads} threads")
            
    async def demo_fallback_configuration(self):
        """Demonstrate progressive fallback configuration"""
        logger.info("\n🔄 Progressive Fallback Configuration:")
        logger.info("-" * 40)
        
        quality_levels = self.downloader.quality_levels
        
        for i, level in enumerate(quality_levels):
            quality = level["quality"]
            memory_mb = level["max_memory_mb"]
            timeout_mult = level["timeout_multiplier"]
            threads = self.downloader._get_optimal_threads(quality, memory_mb)
            
            logger.info(f"Level {i+1}: {quality}")
            logger.info(f"  Memory Limit: {memory_mb} MB")
            logger.info(f"  Timeout Multiplier: {timeout_mult}x")
            logger.info(f"  Optimal Threads: {threads}")
            logger.info(f"  Base Timeout: {int(30 * 60 * timeout_mult / 60)} minutes")
            
            if i < len(quality_levels) - 1:
                logger.info("  ↓ Falls back to next level on memory/timeout errors")
            else:
                logger.info("  ⚠️ Final fallback level")
            logger.info("")
            
    async def demo_memory_scenarios(self):
        """Demonstrate different memory scenarios"""
        logger.info("\n💾 Memory Scenario Analysis:")
        logger.info("-" * 40)
        
        scenarios = [
            (512, "Very constrained (512MB)"),
            (1024, "Constrained (1GB)"),
            (2048, "Limited (2GB)"),
            (4096, "Moderate (4GB)"),
            (8192, "Comfortable (8GB)"),
            (16384, "High-end (16GB)"),
        ]
        
        for memory_mb, description in scenarios:
            recommended = self.downloader.get_quality_recommendation(memory_mb)
            threads = self.downloader._get_optimal_threads(recommended, memory_mb)
            
            # Find the quality level
            quality_level = None
            for i, level in enumerate(self.downloader.quality_levels):
                if level["quality"] == recommended:
                    quality_level = i + 1
                    timeout_mult = level["timeout_multiplier"]
                    break
            
            logger.info(f"{description}:")
            logger.info(f"  Recommended Quality: {recommended}")
            logger.info(f"  Fallback Level: {quality_level}/4")
            logger.info(f"  Optimal Threads: {threads}")
            logger.info(f"  Timeout: {int(30 * timeout_mult)} minutes")
            
            # Show what happens with long videos
            long_video_quality = self.downloader.get_quality_recommendation(memory_mb, 240)  # 4 hours
            if long_video_quality != recommended:
                logger.info(f"  For 4+ hour videos: {long_video_quality} (downgraded)")
            
            logger.info("")

async def main():
    """Main demo function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Quality Fallback Demo")
        print("Usage: python demo_quality_fallback.py")
        print("\nThis script demonstrates the progressive quality fallback system.")
        return
    
    demo = QualityFallbackDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
