import time
import random
import asyncio
import threading
import logging
import statistics
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Add parent directory to path to import the message queue module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import message queue components
from message_queue import (
    Message, 
    PriorityMessageQueue, 
    QueueMetrics, 
    MessageHandler, 
    MessageProducer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stress_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("StressTest")

@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    queue_size: int = 100000
    worker_count: int = 40
    message_count: int = 1000000
    producer_threads: int = 20
    max_retries: int = 3
    test_duration_seconds: int = 120
    message_size_bytes: int = 1024
    failure_rate: float = 0.05
    processing_time_min: float = 0.001
    processing_time_max: float = 0.05
    priority_distribution: Dict[int, float] = field(default_factory=lambda: {
        0: 0.1,  # 10% high priority
        1: 0.2,  # 20% medium-high priority
        2: 0.3,  # 30% medium priority
        3: 0.4,  # 40% low priority
    })
    message_types: List[str] = field(default_factory=lambda: [
        "standard", "batch", "critical", "slow", "bulk"
    ])
    bursts_enabled: bool = True
    burst_size: int = 10000
    burst_interval_seconds: int = 10
    enable_backpressure_test: bool = True
    report_interval_seconds: float = 1.0
    warmup_time_seconds: float = 5.0

@dataclass
class TestResults:
    """Results from stress testing"""
    messages_sent: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    latencies: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    queue_sizes: List[int] = field(default_factory=list)
    backpressure_events: int = 0
    processing_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from the test results"""
        duration = self.end_time - self.start_time
        
        # Handle empty lists
        latency_stats = {
            "min": min(self.latencies) if self.latencies else 0,
            "max": max(self.latencies) if self.latencies else 0,
            "avg": statistics.mean(self.latencies) if self.latencies else 0,
            "median": statistics.median(self.latencies) if self.latencies else 0,
            "p95": statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else 0,
            "p99": statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) >= 100 else 0,
        }
        
        throughput_stats = {
            "min": min(self.throughputs) if self.throughputs else 0,
            "max": max(self.throughputs) if self.throughputs else 0,
            "avg": statistics.mean(self.throughputs) if self.throughputs else 0,
        }
        
        queue_stats = {
            "min": min(self.queue_sizes) if self.queue_sizes else 0,
            "max": max(self.queue_sizes) if self.queue_sizes else 0,
            "avg": statistics.mean(self.queue_sizes) if self.queue_sizes else 0,
        }
        
        processing_stats = {
            "min": min(self.processing_times) if self.processing_times else 0,
            "max": max(self.processing_times) if self.processing_times else 0,
            "avg": statistics.mean(self.processing_times) if self.processing_times else 0,
        }
        
        return {
            "messages_sent": self.messages_sent,
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "duration_seconds": duration,
            "overall_throughput": self.messages_processed / duration if duration > 0 else 0,
            "success_rate": (self.messages_processed / self.messages_sent) * 100 if self.messages_sent > 0 else 0,
            "latency": latency_stats,
            "throughput": throughput_stats,
            "queue_size": queue_stats,
            "processing_time": processing_stats,
            "backpressure_events": self.backpressure_events,
            "error_count": len(self.errors),
        }

class MessageTracker:
    """Tracks message latencies and status"""
    
    def __init__(self):
        self.send_times: Dict[str, float] = {}
        self.latencies: Dict[str, float] = {}
        self.mutex = threading.Lock()
        
    def record_send(self, message_id: str):
        """Record when a message was sent"""
        with self.mutex:
            self.send_times[message_id] = time.time()
    
    def record_completion(self, message_id: str) -> float:
        """Record when a message was completed and return latency"""
        with self.mutex:
            if message_id in self.send_times:
                latency = time.time() - self.send_times[message_id]
                self.latencies[message_id] = latency
                return latency
            return 0.0
    
    def get_latencies(self) -> List[float]:
        """Get all recorded latencies"""
        with self.mutex:
            return list(self.latencies.values())
    
    def cleanup(self, max_age: float = 60.0):
        """Remove old entries to prevent memory leaks"""
        now = time.time()
        with self.mutex:
            # Remove old send times
            to_remove = []
            for msg_id, send_time in self.send_times.items():
                if now - send_time > max_age:
                    to_remove.append(msg_id)
            
            for msg_id in to_remove:
                self.send_times.pop(msg_id, None)
                
            # Limit latencies history size
            if len(self.latencies) > 100000:
                oldest = sorted(self.latencies.items(), key=lambda x: x[1])[:50000]
                for msg_id, _ in oldest:
                    self.latencies.pop(msg_id, None)

class StressTest:
    """Stress test for message queue system"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results = TestResults()
        self.running = False
        self.tracker = MessageTracker()
        
        # Create components
        self.queue = PriorityMessageQueue(max_size=config.queue_size)
        self.producer = MessageProducer(self.queue)
        self.handler = MessageHandler(
            queue=self.queue,
            worker_count=config.worker_count,
            max_retries=config.max_retries
        )
        
        # Internal state
        self.last_report_time = 0.0
        self.last_messages_processed = 0
        self.last_cleanup_time = 0.0
        self.throughput_report_interval = 1.0  # seconds
        
        # Configure processors
        self._register_processors()
        
    def _register_processors(self):
        """Register message processors based on types"""
        # Standard processor
        self.handler.register_processor("standard", self._create_processor())
        
        # Critical processor (faster)
        self.handler.register_processor("critical", self._create_processor(
            min_time=self.config.processing_time_min / 2,
            max_time=self.config.processing_time_max / 2,
            failure_rate=self.config.failure_rate / 2
        ))
        
        # Slow processor
        self.handler.register_processor("slow", self._create_processor(
            min_time=self.config.processing_time_max,
            max_time=self.config.processing_time_max * 3,
            failure_rate=self.config.failure_rate * 1.5
        ))
        
        # Batch processor
        self.handler.register_processor("batch", self._create_processor())
        
        # Bulk processor (very fast)
        self.handler.register_processor("bulk", self._create_processor(
            min_time=self.config.processing_time_min / 4,
            max_time=self.config.processing_time_min * 2,
            failure_rate=self.config.failure_rate / 4
        ))
        
        # Default fallback
        self.handler.register_processor("default", self._create_processor())
    
    def _create_processor(self, min_time=None, max_time=None, failure_rate=None):
        """Create a message processor with configurable parameters"""
        if min_time is None:
            min_time = self.config.processing_time_min
        if max_time is None:
            max_time = self.config.processing_time_max
        if failure_rate is None:
            failure_rate = self.config.failure_rate
            
        async def processor(message: Message) -> bool:
            # Track processing time
            start_time = time.time()
            
            # Simulate processing work
            process_time = random.uniform(min_time, max_time)
            await asyncio.sleep(process_time)
            
            # Record completion and latency
            latency = self.tracker.record_completion(message.id)
            
            # Record processing time
            with threading.Lock():
                self.results.processing_times.append(process_time)
            
            # Simulate occasional failures
            if random.random() < failure_rate:
                with threading.Lock():
                    self.results.messages_failed += 1
                return False
            
            # Success
            with threading.Lock():
                self.results.messages_processed += 1
            return True
            
        return processor
    
    def _generate_message_payload(self, index: int) -> Dict[str, Any]:
        """Generate a message payload of specified size"""
        message_type = random.choice(self.config.message_types)
        
        # Generate randomized payload to prevent compression optimization
        random_data = ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for _ in range(self.config.message_size_bytes)
        )
        
        return {
            "type": message_type,
            "index": index,
            "timestamp": time.time(),
            "data": random_data[:self.config.message_size_bytes]
        }
    
    def _get_random_priority(self) -> int:
        """Get a random priority based on the configured distribution"""
        r = random.random()
        cumulative = 0.0
        
        for priority, probability in self.config.priority_distribution.items():
            cumulative += probability
            if r <= cumulative:
                return priority
                
        # Fallback to lowest priority
        return max(self.config.priority_distribution.keys())
    
    def _producer_worker(self, thread_id: int, stop_event: threading.Event):
        """Worker thread for producing messages"""
        message_index = thread_id * 1000000  # Use thread ID to ensure unique message indices
        send_count = 0
        
        try:
            while not stop_event.is_set() and (
                self.config.message_count <= 0 or  # Unlimited messages
                self.results.messages_sent < self.config.message_count  # Limited messages
            ):
                try:
                    # Generate message
                    payload = self._generate_message_payload(message_index)
                    priority = self._get_random_priority()
                    
                    # Send message
                    message_id = self.producer.send_message(payload, priority)
                    
                    # Track message
                    self.tracker.record_send(message_id)
                    
                    # Update counters
                    with threading.Lock():
                        self.results.messages_sent += 1
                    
                    message_index += 1
                    send_count += 1
                    
                    # Add small delay to simulate realistic production patterns
                    if send_count % 100 == 0:
                        time.sleep(0.001)
                        
                except Exception as e:
                    with threading.Lock():
                        self.results.errors.append(f"Producer error: {str(e)}")
                    logger.error(f"Producer {thread_id} error: {str(e)}")
                    time.sleep(0.1)  # Back off on error
        except Exception as e:
            logger.error(f"Producer thread {thread_id} crashed: {str(e)}")
    
    def _burst_worker(self, stop_event: threading.Event):
        """Worker that periodically sends bursts of messages"""
        while not stop_event.is_set():
            try:
                # Sleep until next burst
                time.sleep(self.config.burst_interval_seconds)
                
                if stop_event.is_set():
                    break
                
                # Send a burst of messages
                logger.info(f"Sending burst of {self.config.burst_size} messages")
                
                payloads = []
                for i in range(self.config.burst_size):
                    payload = self._generate_message_payload(i)
                    payloads.append(payload)
                    
                # Use batch send for efficiency
                message_ids = self.producer.send_batch(payloads, priority=0)  # High priority
                
                # Track messages
                for message_id in message_ids:
                    self.tracker.record_send(message_id)
                
                # Update counter
                with threading.Lock():
                    self.results.messages_sent += len(message_ids)
                
                logger.info(f"Burst complete, sent {len(message_ids)} messages")
                
            except Exception as e:
                with threading.Lock():
                    self.results.errors.append(f"Burst error: {str(e)}")
                logger.error(f"Burst worker error: {str(e)}")
    
    def _backpressure_tester(self, stop_event: threading.Event):
        """Tests system behavior under backpressure conditions"""
        if not self.config.enable_backpressure_test:
            return
            
        # Wait for warmup
        time.sleep(self.config.warmup_time_seconds)
        
        while not stop_event.is_set():
            try:
                # Attempt to overload the queue
                logger.info("Starting backpressure test")
                
                # Generate large number of messages quickly
                payloads = []
                for i in range(self.config.queue_size * 2):  # Try to send twice the queue capacity
                    if stop_event.is_set() or self.queue.size() >= self.config.queue_size * 0.9:
                        break
                        
                    payload = self._generate_message_payload(i)
                    payloads.append(payload)
                    
                    if i % 1000 == 0:
                        # Send in batches of 1000
                        message_ids = self.producer.send_batch(payloads, priority=3)  # Low priority
                        
                        for message_id in message_ids:
                            self.tracker.record_send(message_id)
                            
                        with threading.Lock():
                            self.results.messages_sent += len(message_ids)
                            
                        payloads = []
                
                # Send any remaining payloads
                if payloads:
                    message_ids = self.producer.send_batch(payloads, priority=3)
                    
                    for message_id in message_ids:
                        self.tracker.record_send(message_id)
                        
                    with threading.Lock():
                        self.results.messages_sent += len(message_ids)
                
                # Track backpressure event
                with threading.Lock():
                    self.results.backpressure_events += 1
                    
                logger.info(f"Backpressure test complete, queue size: {self.queue.size()}")
                
                # Wait before next backpressure test
                time.sleep(self.config.burst_interval_seconds * 3)
                
            except Exception as e:
                with threading.Lock():
                    self.results.errors.append(f"Backpressure error: {str(e)}")
                logger.error(f"Backpressure tester error: {str(e)}")
                time.sleep(1.0)  # Back off on error
    
    def _monitor_worker(self, stop_event: threading.Event):
        """Monitors and reports on system performance"""
        self.last_report_time = time.time()
        self.last_messages_processed = 0
        self.last_cleanup_time = time.time()
        
        while not stop_event.is_set():
            try:
                # Sleep briefly
                time.sleep(self.config.report_interval_seconds)
                
                # Get current time
                current_time = time.time()
                
                # Get metrics
                metrics = self.handler.get_metrics()
                queue_size = self.queue.size()
                
                # Calculate throughput
                time_delta = current_time - self.last_report_time
                message_delta = metrics["processed_messages"] - self.last_messages_processed
                
                if time_delta > 0:
                    throughput = message_delta / time_delta
                    
                    # Record throughput and queue size
                    with threading.Lock():
                        self.results.throughputs.append(throughput)
                        self.results.queue_sizes.append(queue_size)
                    
                    # Update last values
                    self.last_report_time = current_time
                    self.last_messages_processed = metrics["processed_messages"]
                    
                    # Get latencies
                    latencies = self.tracker.get_latencies()
                    
                    # Record latencies
                    with threading.Lock():
                        self.results.latencies.extend(latencies)
                    
                    # Log status
                    logger.info(
                        f"Status: Sent={self.results.messages_sent}, "
                        f"Processed={metrics['processed_messages']}, "
                        f"Failed={metrics['failed_messages']}, "
                        f"Queue Size={queue_size}, "
                        f"Throughput={throughput:.2f} msgs/sec, "
                        f"Avg Latency={statistics.mean(latencies) if latencies else 0:.6f}s"
                    )
                
                # Periodic cleanup to prevent memory issues
                if current_time - self.last_cleanup_time > 10.0:
                    self.tracker.cleanup()
                    self.last_cleanup_time = current_time
                
            except Exception as e:
                logger.error(f"Monitor error: {str(e)}")
    
    def run(self):
        """Run the stress test"""
        logger.info("Starting stress test with configuration:")
        for key, value in vars(self.config).items():
            logger.info(f"  {key}: {value}")
        
        self.results = TestResults()
        self.results.start_time = time.time()
        
        # Start message handler
        self.handler.start()
        logger.info(f"Started message handler with {self.config.worker_count} workers")
        
        # Create stop event
        stop_event = threading.Event()
        
        try:
            # Start producer threads
            producer_threads = []
            for i in range(self.config.producer_threads):
                thread = threading.Thread(
                    target=self._producer_worker,
                    args=(i, stop_event),
                    name=f"producer-{i}"
                )
                thread.daemon = True
                thread.start()
                producer_threads.append(thread)
            
            logger.info(f"Started {self.config.producer_threads} producer threads")
            
            # Start burst worker if enabled
            burst_thread = None
            if self.config.bursts_enabled:
                burst_thread = threading.Thread(
                    target=self._burst_worker,
                    args=(stop_event,),
                    name="burst-worker"
                )
                burst_thread.daemon = True
                burst_thread.start()
                logger.info("Started burst worker")
            
            # Start backpressure tester if enabled
            backpressure_thread = None
            if self.config.enable_backpressure_test:
                backpressure_thread = threading.Thread(
                    target=self._backpressure_tester,
                    args=(stop_event,),
                    name="backpressure-tester"
                )
                backpressure_thread.daemon = True
                backpressure_thread.start()
                logger.info("Started backpressure tester")
            
            # Start monitor thread
            monitor_thread = threading.Thread(
                target=self._monitor_worker,
                args=(stop_event,),
                name="monitor"
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            logger.info("Started monitor thread")
            
            # Run for specified duration
            logger.info(f"Test running for {self.config.test_duration_seconds} seconds")
            test_start_time = time.time()
            
            try:
                while (time.time() - test_start_time) < self.config.test_duration_seconds:
                    time.sleep(0.1)
                    
                    # Check if reached message limit
                    if (self.config.message_count > 0 and 
                        self.results.messages_processed >= self.config.message_count):
                        logger.info(f"Reached message count limit of {self.config.message_count}")
                        break
            except KeyboardInterrupt:
                logger.info("Test interrupted by user")
            
            # Signal threads to stop
            stop_event.set()
            logger.info("Signaled threads to stop")
            
            # Wait for completion with timeout
            completion_timeout = 10.0
            completion_start = time.time()
            
            while (time.time() - completion_start) < completion_timeout:
                if self.queue.size() == 0:
                    break
                time.sleep(0.1)
            
            # Set end time
            self.results.end_time = time.time()
            
            # Collect final latencies
            final_latencies = self.tracker.get_latencies()
            with threading.Lock():
                self.results.latencies.extend(final_latencies)
            
            # Get final metrics
            final_metrics = self.handler.get_metrics()
            self.results.messages_processed = final_metrics["processed_messages"]
            self.results.messages_failed = final_metrics["failed_messages"]
            
            # Calculate statistics
            stats = self.results.calculate_statistics()
            
            # Log results
            logger.info("Stress test completed")
            logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
            logger.info(f"Messages sent: {stats['messages_sent']}")
            logger.info(f"Messages processed: {stats['messages_processed']}")
            logger.info(f"Messages failed: {stats['messages_failed']}")
            logger.info(f"Success rate: {stats['success_rate']:.2f}%")
            logger.info(f"Overall throughput: {stats['overall_throughput']:.2f} msgs/sec")
            logger.info(f"Average latency: {stats['latency']['avg']:.6f} seconds")
            logger.info(f"95th percentile latency: {stats['latency']['p95']:.6f} seconds")
            logger.info(f"Maximum latency: {stats['latency']['max']:.6f} seconds")
            logger.info(f"Maximum queue size: {stats['queue_size']['max']}")
            logger.info(f"Backpressure events: {stats['backpressure_events']}")
            logger.info(f"Errors: {stats['error_count']}")
            
            # Print detailed stats
            self._print_detailed_stats(stats)
            
            return stats
            
        finally:
            # Stop threads
            stop_event.set()
            
            # Stop handler
            self.handler.stop()
            logger.info("Stopped message handler")
    
    def _print_detailed_stats(self, stats: Dict[str, Any]):
        """Print detailed statistics"""
        print("\n==== STRESS TEST RESULTS ====")
        print(f"Test Duration: {stats['duration_seconds']:.2f} seconds")
        
        print("\n-- Message Counts --")
        print(f"Sent:      {stats['messages_sent']:,}")
        print(f"Processed: {stats['messages_processed']:,}")
        print(f"Failed:    {stats['messages_failed']:,}")
        print(f"Success:   {stats['success_rate']:.2f}%")
        
        print("\n-- Performance --")
        print(f"Overall Throughput: {stats['overall_throughput']:,.2f} msgs/sec")
        print(f"Peak Throughput:    {stats['throughput']['max']:,.2f} msgs/sec")
        print(f"Average Throughput: {stats['throughput']['avg']:,.2f} msgs/sec")
        
        print("\n-- Latency (seconds) --")
        print(f"Average:  {stats['latency']['avg']:.6f}")
        print(f"Median:   {stats['latency']['median']:.6f}")
        print(f"95th Pct: {stats['latency']['p95']:.6f}")
        print(f"99th Pct: {stats['latency']['p99']:.6f}")
        print(f"Min:      {stats['latency']['min']:.6f}")
        print(f"Max:      {stats['latency']['max']:.6f}")
        
        print("\n-- Queue Stats --")
        print(f"Max Size: {stats['queue_size']['max']:,}")
        print(f"Avg Size: {stats['queue_size']['avg']:,.2f}")
        
        print("\n-- Processing Time (seconds) --")
        print(f"Average: {stats['processing_time']['avg']:.6f}")
        print(f"Min:     {stats['processing_time']['min']:.6f}")
        print(f"Max:     {stats['processing_time']['max']:.6f}")
        
        print("\n-- Other Stats --")
        print(f"Backpressure Events: {stats['backpressure_events']}")
        print(f"Errors:              {stats['error_count']}")
        print("=============================\n")

def main():
    """Main entry point for stress test"""
    parser = argparse.ArgumentParser(description="Message Queue Stress Test")
    
    parser.add_argument("--queue-size", type=int, default=100000,
                       help="Maximum queue size (default: 100000)")
    parser.add_argument("--workers", type=int, default=40,
                       help="Number of worker threads (default: 40)")
    parser.add_argument("--producers", type=int, default=20,
                       help="Number of producer threads (default: 20)")
    parser.add_argument("--duration", type=int, default=120,
                       help="Test duration in seconds (default: 120)")
    parser.add_argument("--message-size", type=int, default=1024,
                       help="Message size in bytes (default: 1024)")
    parser.add_argument("--no-bursts", action="store_true",
                       help="Disable burst testing")
    parser.add_argument("--no-backpressure", action="store_true",
                       help="Disable backpressure testing")
    parser.add_argument("--failure-rate", type=float, default=0.05,
                       help="Processor failure rate (default: 0.05)")
    
    args = parser.parse_args()
    
    # Create config
    config = StressTestConfig(
        queue_size=args.queue_size,
        worker_count=args.workers,
        producer_threads=args.producers,
        test_duration_seconds=args.duration,
        message_size_bytes=args.message_size,
        bursts_enabled=not args.no_bursts,
        enable_backpressure_test=not args.no_backpressure,
        failure_rate=args.failure_rate
    )
    
    # Run test
    test = StressTest(config)
    test.run()

if __name__ == "__main__":
    main()