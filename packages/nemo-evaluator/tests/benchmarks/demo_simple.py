#!/usr/bin/env python3
"""
Simple demonstration of how the profiling works.
Run this to understand the core concept!

Usage:
    python demo_simple.py
"""

import time
from contextlib import contextmanager


# ============================================================================
# Step 1: Create a simple profiler (like a stopwatch)
# ============================================================================
class SimpleProfiler:
    """A stopwatch that measures how long code takes to run."""
    
    def __init__(self):
        self.timings = {}  # Store all measurements
    
    @contextmanager
    def measure(self, operation_name):
        """
        Context manager that acts like a stopwatch.
        
        When you write:
            with profiler.measure("my_operation"):
                do_something()
        
        It automatically:
        1. Starts a timer before do_something()
        2. Lets your code run
        3. Stops the timer after do_something()
        4. Saves how long it took
        """
        print(f"  ⏱️  Starting timer for: {operation_name}")
        start_time = time.perf_counter()  # Start stopwatch
        
        try:
            yield  # Your code runs here
        finally:
            elapsed_time = time.perf_counter() - start_time  # Stop stopwatch
            
            # Save the measurement
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed_time)
            
            print(f"  ⏹️  Stopped timer for: {operation_name} → {elapsed_time*1000:.3f}ms")
    
    def print_summary(self):
        """Show all measurements in a nice table."""
        print("\n" + "="*70)
        print("PROFILING RESULTS")
        print("="*70)
        print(f"{'Operation':<30} {'Total (ms)':<15} {'Calls':<10} {'Avg (ms)'}")
        print("-"*70)
        
        for operation, times in self.timings.items():
            total = sum(times) * 1000  # Convert to milliseconds
            count = len(times)
            avg = total / count
            print(f"{operation:<30} {total:<15.3f} {count:<10} {avg:.3f}")
        
        print("="*70)


# ============================================================================
# Step 2: Original code (without profiling)
# ============================================================================
class OriginalWorker:
    """This is like the original ResponseStatsInterceptor - fast but no visibility."""
    
    def do_work(self, task_number):
        """Process a task - but we can't see where time is spent!"""
        self._step1_fast()
        self._step2_slow()
        self._step3_medium()
        return f"Task {task_number} complete"
    
    def _step1_fast(self):
        """A fast operation."""
        x = sum(range(1000))
    
    def _step2_slow(self):
        """A slow operation (simulating disk I/O)."""
        time.sleep(0.01)  # Simulate slow disk write
    
    def _step3_medium(self):
        """A medium operation."""
        time.sleep(0.003)


# ============================================================================
# Step 3: Instrumented code (with profiling)
# ============================================================================
class InstrumentedWorker(OriginalWorker):
    """
    This wraps the original code with timing measurements.
    Like InstrumentedResponseStatsInterceptor wraps ResponseStatsInterceptor.
    """
    
    def __init__(self, profiler):
        self.profiler = profiler
        super().__init__()
    
    def do_work(self, task_number):
        """Now we measure the total time!"""
        with self.profiler.measure("do_work_TOTAL"):
            return super().do_work(task_number)
    
    def _step1_fast(self):
        """Measure this step."""
        with self.profiler.measure("_step1_fast"):
            super()._step1_fast()
    
    def _step2_slow(self):
        """Measure this step."""
        with self.profiler.measure("_step2_slow"):
            super()._step2_slow()
    
    def _step3_medium(self):
        """Measure this step."""
        with self.profiler.measure("_step3_medium"):
            super()._step3_medium()


# ============================================================================
# Demonstration
# ============================================================================
def main():
    print("\n" + "="*70)
    print("DEMONSTRATION: How Profiling Works")
    print("="*70)
    
    # Create profiler
    profiler = SimpleProfiler()
    
    # Create instrumented worker
    worker = InstrumentedWorker(profiler)
    
    print("\nProcessing 5 tasks...\n")
    
    # Process multiple tasks
    for i in range(5):
        print(f"\n--- Task {i+1} ---")
        result = worker.do_work(i+1)
        print(f"✅ {result}")
    
    # Show results
    profiler.print_summary()
    
    print("\n💡 KEY INSIGHT:")
    print("   _step2_slow takes the most time!")
    print("   This is where you should optimize first.")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

