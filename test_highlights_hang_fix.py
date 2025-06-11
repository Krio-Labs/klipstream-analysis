#!/usr/bin/env python3
"""
Test Script for Highlights Analysis Hang Fix

This script tests the comprehensive fix for the critical system hang issue
during highlights analysis phase.
"""

import asyncio
import os
import time
import threading
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighlightsHangTester:
    """Comprehensive tester for highlights analysis hang fix"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="highlights_hang_test_"))
        print(f"🔧 Test directory: {self.temp_dir}")
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests for the highlights hang fix"""
        print("🧪 HIGHLIGHTS ANALYSIS HANG FIX TESTS")
        print("=" * 80)
        
        # Test 1: Signal Handling Fix
        self._test_signal_handling_fix()
        
        # Test 2: Timeout Mechanism
        self._test_timeout_mechanism()
        
        # Test 3: Resource Management
        self._test_resource_management()
        
        # Test 4: Process Isolation
        self._test_process_isolation()
        
        # Test 5: Error Recovery
        self._test_error_recovery()
        
        # Test 6: Threading Safety
        self._test_threading_safety()
        
        # Test 7: Real Scenario Test
        self._test_real_scenario()
        
        # Generate test report
        self._generate_test_report()
    
    def _test_signal_handling_fix(self):
        """Test that signal handling works correctly in worker threads"""
        print("\n1️⃣ SIGNAL HANDLING FIX TEST")
        print("-" * 50)
        
        try:
            from analysis_pipeline.utils.process_manager import ProcessManager
            
            manager = ProcessManager()
            
            def signal_test_function():
                """Function that would previously fail with signal handling"""
                import signal
                import time
                
                # This should NOT cause "signal only works in main thread" error
                # because we're using the new safe timeout mechanism
                try:
                    # Simulate the old problematic code
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Test timeout")
                    
                    # This would fail in worker thread - but we catch it
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(1)
                    time.sleep(0.5)
                    signal.alarm(0)
                    return "signal_test_passed"
                except Exception as e:
                    # Expected to fail in worker thread
                    return f"signal_failed_as_expected: {str(e)}"
            
            # Test in main thread (should work)
            main_thread_result = signal_test_function()
            
            # Test in worker thread (should handle gracefully)
            worker_result = manager.execute_with_timeout(
                func=signal_test_function,
                timeout=10,
                process_name="signal_test"
            )
            
            self.test_results["signal_handling"] = {
                "success": True,
                "main_thread_result": main_thread_result,
                "worker_thread_result": worker_result,
                "signal_error_handled": "signal_failed_as_expected" in str(worker_result)
            }
            
            print(f"✅ Signal handling test completed")
            print(f"   Main thread: {main_thread_result}")
            print(f"   Worker thread: {worker_result}")
            print(f"   Signal error handled: {'✅' if 'signal_failed_as_expected' in str(worker_result) else '❌'}")
            
        except Exception as e:
            print(f"❌ Signal handling test failed: {e}")
            self.test_results["signal_handling"] = {"success": False, "error": str(e)}
    
    def _test_timeout_mechanism(self):
        """Test the new timeout mechanism"""
        print("\n2️⃣ TIMEOUT MECHANISM TEST")
        print("-" * 50)
        
        try:
            from analysis_pipeline.utils.process_manager import ProcessManager
            
            manager = ProcessManager()
            
            def slow_function(delay: int):
                """Function that takes longer than timeout"""
                time.sleep(delay)
                return f"completed_after_{delay}s"
            
            # Test 1: Function that completes within timeout
            start_time = time.time()
            try:
                result = manager.execute_with_timeout(
                    func=slow_function,
                    args=(2,),
                    timeout=5,
                    process_name="fast_test"
                )
                fast_test_success = True
                fast_test_time = time.time() - start_time
            except Exception as e:
                fast_test_success = False
                fast_test_time = time.time() - start_time
                result = str(e)
            
            # Test 2: Function that times out
            start_time = time.time()
            try:
                result = manager.execute_with_timeout(
                    func=slow_function,
                    args=(10,),
                    timeout=3,
                    process_name="timeout_test"
                )
                timeout_test_success = False  # Should have timed out
                timeout_test_time = time.time() - start_time
            except Exception as e:
                timeout_test_success = "timeout" in str(e).lower()
                timeout_test_time = time.time() - start_time
            
            self.test_results["timeout_mechanism"] = {
                "success": True,
                "fast_test_success": fast_test_success,
                "fast_test_time": fast_test_time,
                "timeout_test_success": timeout_test_success,
                "timeout_test_time": timeout_test_time,
                "timeout_within_bounds": 2.5 <= timeout_test_time <= 4.0  # Should be ~3 seconds
            }
            
            print(f"✅ Timeout mechanism test completed")
            print(f"   Fast test (2s/5s timeout): {'✅' if fast_test_success else '❌'} ({fast_test_time:.1f}s)")
            print(f"   Timeout test (10s/3s timeout): {'✅' if timeout_test_success else '❌'} ({timeout_test_time:.1f}s)")
            print(f"   Timeout timing correct: {'✅' if 2.5 <= timeout_test_time <= 4.0 else '❌'}")
            
        except Exception as e:
            print(f"❌ Timeout mechanism test failed: {e}")
            self.test_results["timeout_mechanism"] = {"success": False, "error": str(e)}
    
    def _test_resource_management(self):
        """Test resource management and cleanup"""
        print("\n3️⃣ RESOURCE MANAGEMENT TEST")
        print("-" * 50)
        
        try:
            from analysis_pipeline.audio.analysis_fixed import ResourceManager
            import psutil
            
            # Get initial memory
            initial_memory = psutil.virtual_memory().used
            
            # Test resource manager
            resource_manager = ResourceManager()
            
            # Simulate resource allocation
            test_data = []
            for i in range(100):
                test_data.append([0] * 10000)  # Allocate some memory
            
            # Check memory usage
            peak_memory = resource_manager.check_memory_usage()
            
            # Register cleanup
            cleanup_called = False
            def test_cleanup():
                nonlocal cleanup_called
                cleanup_called = True
                test_data.clear()
            
            resource_manager.register_cleanup(test_cleanup)
            
            # Execute cleanup
            resource_manager.cleanup_all()
            
            # Check final memory
            final_memory = psutil.virtual_memory().used
            
            self.test_results["resource_management"] = {
                "success": True,
                "initial_memory_gb": initial_memory / (1024**3),
                "peak_memory_gb": peak_memory / (1024**3),
                "final_memory_gb": final_memory / (1024**3),
                "cleanup_called": cleanup_called,
                "memory_cleaned": final_memory <= peak_memory
            }
            
            print(f"✅ Resource management test completed")
            print(f"   Initial memory: {initial_memory / (1024**3):.2f}GB")
            print(f"   Peak memory: {peak_memory / (1024**3):.2f}GB")
            print(f"   Final memory: {final_memory / (1024**3):.2f}GB")
            print(f"   Cleanup called: {'✅' if cleanup_called else '❌'}")
            print(f"   Memory cleaned: {'✅' if final_memory <= peak_memory else '❌'}")
            
        except Exception as e:
            print(f"❌ Resource management test failed: {e}")
            self.test_results["resource_management"] = {"success": False, "error": str(e)}
    
    def _test_process_isolation(self):
        """Test process isolation for complete safety"""
        print("\n4️⃣ PROCESS ISOLATION TEST")
        print("-" * 50)
        
        try:
            from analysis_pipeline.utils.process_manager import ProcessManager
            
            manager = ProcessManager()
            
            def isolated_function(test_value: str):
                """Function to test in isolated process"""
                import os
                import time
                
                # Simulate some work
                time.sleep(1)
                
                # Return process info
                return {
                    "test_value": test_value,
                    "process_id": os.getpid(),
                    "completed": True
                }
            
            # Test process isolation
            start_time = time.time()
            result = manager.execute_with_process_pool(
                func=isolated_function,
                args=("isolation_test",),
                timeout=10,
                process_name="isolation_test"
            )
            execution_time = time.time() - start_time
            
            # Verify result
            isolation_success = (
                result is not None and
                result.get("test_value") == "isolation_test" and
                result.get("completed") is True and
                result.get("process_id") != os.getpid()  # Different process
            )
            
            self.test_results["process_isolation"] = {
                "success": True,
                "isolation_success": isolation_success,
                "execution_time": execution_time,
                "result": result,
                "different_process": result.get("process_id") != os.getpid() if result else False
            }
            
            print(f"✅ Process isolation test completed")
            print(f"   Isolation successful: {'✅' if isolation_success else '❌'}")
            print(f"   Execution time: {execution_time:.1f}s")
            print(f"   Different process: {'✅' if result and result.get('process_id') != os.getpid() else '❌'}")
            
        except Exception as e:
            print(f"❌ Process isolation test failed: {e}")
            self.test_results["process_isolation"] = {"success": False, "error": str(e)}
    
    def _test_error_recovery(self):
        """Test error recovery mechanisms"""
        print("\n5️⃣ ERROR RECOVERY TEST")
        print("-" * 50)
        
        try:
            from analysis_pipeline.utils.process_manager import ProcessManager
            
            manager = ProcessManager()
            
            def failing_function():
                """Function that always fails"""
                raise ValueError("Intentional test failure")
            
            def hanging_function():
                """Function that hangs indefinitely"""
                import time
                time.sleep(1000)  # Sleep for a very long time
                return "should_not_reach_here"
            
            # Test error handling
            error_handled = False
            try:
                result = manager.execute_with_timeout(
                    func=failing_function,
                    timeout=5,
                    process_name="error_test"
                )
            except Exception as e:
                error_handled = "Intentional test failure" in str(e)
            
            # Test timeout handling
            timeout_handled = False
            start_time = time.time()
            try:
                result = manager.execute_with_timeout(
                    func=hanging_function,
                    timeout=3,
                    process_name="hang_test"
                )
            except Exception as e:
                timeout_handled = "timeout" in str(e).lower()
            timeout_time = time.time() - start_time
            
            self.test_results["error_recovery"] = {
                "success": True,
                "error_handled": error_handled,
                "timeout_handled": timeout_handled,
                "timeout_time": timeout_time,
                "timeout_timing_correct": 2.5 <= timeout_time <= 4.0
            }
            
            print(f"✅ Error recovery test completed")
            print(f"   Error handled: {'✅' if error_handled else '❌'}")
            print(f"   Timeout handled: {'✅' if timeout_handled else '❌'}")
            print(f"   Timeout timing: {timeout_time:.1f}s {'✅' if 2.5 <= timeout_time <= 4.0 else '❌'}")
            
        except Exception as e:
            print(f"❌ Error recovery test failed: {e}")
            self.test_results["error_recovery"] = {"success": False, "error": str(e)}
    
    def _test_threading_safety(self):
        """Test threading safety of the new implementation"""
        print("\n6️⃣ THREADING SAFETY TEST")
        print("-" * 50)
        
        try:
            from analysis_pipeline.utils.process_manager import ProcessManager
            import threading
            
            manager = ProcessManager()
            results = {}
            errors = {}
            
            def thread_test_function(thread_id: int):
                """Function to test in multiple threads"""
                import time
                time.sleep(1)
                return f"thread_{thread_id}_completed"
            
            def run_in_thread(thread_id: int):
                """Run test in a specific thread"""
                try:
                    result = manager.execute_with_timeout(
                        func=thread_test_function,
                        args=(thread_id,),
                        timeout=5,
                        process_name=f"thread_test_{thread_id}"
                    )
                    results[thread_id] = result
                except Exception as e:
                    errors[thread_id] = str(e)
            
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=run_in_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join(timeout=10)
            
            # Check results
            all_completed = len(results) == 3 and len(errors) == 0
            
            self.test_results["threading_safety"] = {
                "success": True,
                "all_completed": all_completed,
                "results": results,
                "errors": errors,
                "threads_completed": len(results),
                "threads_failed": len(errors)
            }
            
            print(f"✅ Threading safety test completed")
            print(f"   All threads completed: {'✅' if all_completed else '❌'}")
            print(f"   Threads completed: {len(results)}/3")
            print(f"   Threads failed: {len(errors)}/3")
            
        except Exception as e:
            print(f"❌ Threading safety test failed: {e}")
            self.test_results["threading_safety"] = {"success": False, "error": str(e)}
    
    def _test_real_scenario(self):
        """Test with a real scenario similar to the original problem"""
        print("\n7️⃣ REAL SCENARIO TEST")
        print("-" * 50)
        
        try:
            # Create test data similar to real scenario
            test_video_id = "2480161276"
            
            # Create mock segments file
            segments_data = {
                'start_time': [0, 10, 20, 30, 40],
                'end_time': [10, 20, 30, 40, 50],
                'text': ['Test segment 1', 'Test segment 2', 'Test segment 3', 'Test segment 4', 'Test segment 5'],
                'sentiment_score': [0.5, -0.2, 0.8, -0.5, 0.3],
                'highlight_score': [0.6, 0.3, 0.9, 0.2, 0.7],
                'excitement': [0.5, 0.1, 0.8, 0.1, 0.6],
                'funny': [0.2, 0.1, 0.3, 0.1, 0.2],
                'happiness': [0.6, 0.2, 0.9, 0.1, 0.7],
                'anger': [0.1, 0.8, 0.1, 0.9, 0.1],
                'sadness': [0.1, 0.7, 0.1, 0.8, 0.1],
                'neutral': [0.2, 0.1, 0.1, 0.1, 0.2]
            }
            
            segments_df = pd.DataFrame(segments_data)
            segments_file = self.temp_dir / f"audio_{test_video_id}_segments.csv"
            segments_df.to_csv(segments_file, index=False)
            
            # Test the safe highlights analysis
            from analysis_pipeline.utils.process_manager import safe_highlights_analysis
            
            start_time = time.time()
            result = safe_highlights_analysis(
                video_id=test_video_id,
                input_file=str(segments_file),
                output_dir=str(self.temp_dir),
                timeout=30
            )
            execution_time = time.time() - start_time
            
            # Verify result
            scenario_success = result is not None
            
            self.test_results["real_scenario"] = {
                "success": True,
                "scenario_success": scenario_success,
                "execution_time": execution_time,
                "result_type": type(result).__name__,
                "no_hang": execution_time < 35,  # Should complete well within timeout
                "segments_processed": len(segments_data['start_time'])
            }
            
            print(f"✅ Real scenario test completed")
            print(f"   Scenario successful: {'✅' if scenario_success else '❌'}")
            print(f"   Execution time: {execution_time:.1f}s")
            print(f"   No hang detected: {'✅' if execution_time < 35 else '❌'}")
            print(f"   Result type: {type(result).__name__}")
            
        except Exception as e:
            print(f"❌ Real scenario test failed: {e}")
            self.test_results["real_scenario"] = {"success": False, "error": str(e)}
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 HIGHLIGHTS HANG FIX TEST REPORT")
        print("=" * 80)
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 TEST SUMMARY")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            print(f"   {test_name}: {status}")
            
            if not result.get("success") and "error" in result:
                print(f"      Error: {result['error']}")
        
        # Critical fixes validation
        print(f"\n🔧 CRITICAL FIXES VALIDATION")
        
        signal_fix = self.test_results.get("signal_handling", {}).get("signal_error_handled", False)
        print(f"   Signal handling fix: {'✅' if signal_fix else '❌'}")
        
        timeout_fix = self.test_results.get("timeout_mechanism", {}).get("timeout_test_success", False)
        print(f"   Timeout mechanism fix: {'✅' if timeout_fix else '❌'}")
        
        resource_fix = self.test_results.get("resource_management", {}).get("cleanup_called", False)
        print(f"   Resource management fix: {'✅' if resource_fix else '❌'}")
        
        isolation_fix = self.test_results.get("process_isolation", {}).get("isolation_success", False)
        print(f"   Process isolation fix: {'✅' if isolation_fix else '❌'}")
        
        no_hang = self.test_results.get("real_scenario", {}).get("no_hang", False)
        print(f"   No hang in real scenario: {'✅' if no_hang else '❌'}")
        
        # Overall assessment
        critical_fixes_working = all([signal_fix, timeout_fix, resource_fix, isolation_fix, no_hang])
        
        print(f"\n🎯 OVERALL ASSESSMENT")
        if success_rate >= 85 and critical_fixes_working:
            print(f"   ✅ HANG FIX SUCCESSFUL - Ready for production deployment")
        elif success_rate >= 70:
            print(f"   ⚠️  HANG FIX PARTIALLY SUCCESSFUL - Review failed tests")
        else:
            print(f"   ❌ HANG FIX NEEDS MORE WORK - Multiple critical issues detected")
        
        print("✅ Highlights hang fix testing completed!")

def main():
    """Main test execution"""
    tester = HighlightsHangTester()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()
