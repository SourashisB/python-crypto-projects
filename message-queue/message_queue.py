import asyncio
import logging
import time
import uuid
import random
from typing import Dict, List, Any, Callable, Coroutine, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("message_queue.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MessageQueue")

@dataclass
class Message:
    """Message object to be passed through the queue"""
    id: str
    payload: Any
    timestamp: float
    priority: int = 0  # Lower number means higher priority
    
    def __str__(self) -> str:
        return f"Message(id={self.id}, priority={self.priority}, timestamp={self.timestamp})"


class PriorityMessageQueue:
    """Thread-safe priority message queue implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.mutex = threading.Lock()
        self.message_count = 0
        self.max_size = max_size
        
    def put(self, message: Message) -> bool:
        """Put a message in the queue with priority"""
        try:
            with self.mutex:
                if self.message_count >= self.max_size:
                    logger.warning(f"Queue is full, dropping message {message.id}")
                    return False
                
                # Add to priority queue - tuple of (priority, timestamp for FIFO within same priority, message)
                self.queue.put((message.priority, message.timestamp, message))
                self.message_count += 1
                logger.debug(f"Added message {message.id} to queue. Queue size: {self.message_count}")
                return True
        except Exception as e:
            logger.error(f"Error putting message in queue: {e}")
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Message]:
        """Get the next message from the queue based on priority"""
        try:
            with self.mutex:
                if self.message_count <= 0:
                    return None
                
            # Get from priority queue
            priority, _, message = self.queue.get(block=block, timeout=timeout)
            
            with self.mutex:
                self.message_count -= 1
                logger.debug(f"Retrieved message {message.id} from queue. Queue size: {self.message_count}")
            
            return message
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting message from queue: {e}")
            return None
    
    def size(self) -> int:
        """Get current queue size"""
        with self.mutex:
            return self.message_count


class QueueMetrics:
    """Class to track queue metrics"""
    
    def __init__(self):
        self.processed_messages = 0
        self.failed_messages = 0
        self.avg_processing_time = 0
        self.mutex = threading.Lock()
        self.start_time = time.time()
        
    def record_processed(self, processing_time: float):
        """Record a successfully processed message"""
        with self.mutex:
            # Update running average
            self.avg_processing_time = (self.avg_processing_time * self.processed_messages + processing_time) / (self.processed_messages + 1)
            self.processed_messages += 1
    
    def record_failed(self):
        """Record a failed message"""
        with self.mutex:
            self.failed_messages += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.mutex:
            uptime = time.time() - self.start_time
            return {
                "processed_messages": self.processed_messages,
                "failed_messages": self.failed_messages,
                "avg_processing_time": self.avg_processing_time,
                "messages_per_second": self.processed_messages / uptime if uptime > 0 else 0,
                "uptime_seconds": uptime
            }


class MessageHandler:
    """Handles the processing of messages from the queue"""
    
    def __init__(self, 
                 queue: PriorityMessageQueue, 
                 worker_count: int = 20,
                 max_retries: int = 3):
        self.queue = queue
        self.worker_count = worker_count
        self.max_retries = max_retries
        self.running = False
        self.workers: List[threading.Thread] = []
        self.metrics = QueueMetrics()
        self.processors: Dict[str, Callable[[Message], Coroutine[Any, Any, bool]]] = {}
        self.executor = ThreadPoolExecutor(max_workers=worker_count)
        self.retry_queue = PriorityMessageQueue()
        
    def register_processor(self, message_type: str, processor: Callable[[Message], Coroutine[Any, Any, bool]]):
        """Register a processor function for a specific message type"""
        self.processors[message_type] = processor
        logger.info(f"Registered processor for message type: {message_type}")
        
    async def process_message(self, message: Message) -> bool:
        """Process a single message"""
        message_type = message.payload.get("type") if isinstance(message.payload, dict) else "default"
        processor = self.processors.get(message_type, self.processors.get("default"))
        
        if not processor:
            logger.warning(f"No processor found for message type: {message_type}")
            return False
        
        try:
            start_time = time.time()
            result = await processor(message)
            processing_time = time.time() - start_time
            
            if result:
                self.metrics.record_processed(processing_time)
                logger.info(f"Successfully processed message {message.id} in {processing_time:.4f}s")
            else:
                self.metrics.record_failed()
                logger.warning(f"Failed to process message {message.id}")
            
            return result
        except Exception as e:
            self.metrics.record_failed()
            logger.error(f"Error processing message {message.id}: {str(e)}")
            return False
    
    def worker_loop(self):
        """Main worker loop to process messages"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Check for retry messages first
                message = self.retry_queue.get(block=False)
                if not message:
                    # Then check the main queue
                    message = self.queue.get(block=True, timeout=1.0)
                
                if message:
                    retry_count = message.payload.get("retry_count", 0) if isinstance(message.payload, dict) else 0
                    success = loop.run_until_complete(self.process_message(message))
                    
                    # Handle retries if processing failed
                    if not success and retry_count < self.max_retries:
                        if isinstance(message.payload, dict):
                            message.payload["retry_count"] = retry_count + 1
                        else:
                            message.payload = {"data": message.payload, "retry_count": retry_count + 1}
                        
                        # Add a small delay before retry
                        time.sleep(0.1 * (retry_count + 1))
                        self.retry_queue.put(message)
                        logger.info(f"Scheduled message {message.id} for retry {retry_count + 1}/{self.max_retries}")
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
    
    def start(self):
        """Start the message handler workers"""
        if self.running:
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.worker_count):
            worker = threading.Thread(target=self.worker_loop, name=f"worker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {self.worker_count} worker threads")
        
    def stop(self):
        """Stop the message handler"""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            
        self.workers = []
        logger.info("Stopped all worker threads")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self.metrics.get_metrics()
        metrics["queue_size"] = self.queue.size()
        metrics["retry_queue_size"] = self.retry_queue.size()
        metrics["worker_count"] = self.worker_count
        return metrics


class MessageProducer:
    """Produces messages to the queue"""
    
    def __init__(self, queue: PriorityMessageQueue):
        self.queue = queue
        
    def send_message(self, payload: Any, priority: int = 0) -> str:
        """Send a message to the queue"""
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        
        success = self.queue.put(message)
        if success:
            logger.info(f"Sent message {message_id} with priority {priority}")
        else:
            logger.warning(f"Failed to send message {message_id}")
            
        return message_id
    
    def send_batch(self, payloads: List[Any], priority: int = 0) -> List[str]:
        """Send a batch of messages to the queue"""
        message_ids = []
        
        for payload in payloads:
            message_id = self.send_message(payload, priority)
            message_ids.append(message_id)
            
        return message_ids


# Example usage and testing functions

async def sample_processor(message: Message) -> bool:
    """Sample message processor that simulates work"""
    process_time = random.uniform(0.01, 0.2)  # Simulate processing time between 10-200ms
    await asyncio.sleep(process_time)  # Simulate async work
    
    # Simulate occasional failures
    if random.random() < 0.05:  # 5% failure rate
        return False
    
    return True

async def slow_processor(message: Message) -> bool:
    """A slower processor for certain message types"""
    process_time = random.uniform(0.2, 0.5)  # Simulate longer processing time
    await asyncio.sleep(process_time)
    return random.random() < 0.9  # 10% failure rate

def load_test(producer: MessageProducer, num_messages: int = 1000, concurrency: int = 50):
    """Run a load test on the queue"""
    logger.info(f"Starting load test with {num_messages} messages and concurrency {concurrency}")
    
    message_types = ["default", "critical", "batch", "slow"]
    priorities = [0, 1, 2, 3]  # Lower is higher priority
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        
        for i in range(num_messages):
            message_type = random.choice(message_types)
            priority = random.choice(priorities)
            
            payload = {
                "type": message_type,
                "data": f"Message data {i}",
                "timestamp": time.time()
            }
            
            # Submit send_message to executor
            futures.append(executor.submit(producer.send_message, payload, priority))
            
            # Introduce small delay to simulate real-world message arrival pattern
            if i % 10 == 0:
                time.sleep(0.001)
    
    # Wait for all messages to be sent
    for future in futures:
        future.result()
    
    elapsed = time.time() - start_time
    messages_per_second = num_messages / elapsed
    
    logger.info(f"Load test completed: sent {num_messages} messages in {elapsed:.2f} seconds ({messages_per_second:.2f} msgs/sec)")

def print_metrics(handler: MessageHandler):
    """Print current metrics"""
    metrics = handler.get_metrics()
    print("\n=== Queue Metrics ===")
    print(f"Processed: {metrics['processed_messages']}")
    print(f"Failed: {metrics['failed_messages']}")
    print(f"Average processing time: {metrics['avg_processing_time']:.6f} seconds")
    print(f"Messages per second: {metrics['messages_per_second']:.2f}")
    print(f"Queue size: {metrics['queue_size']}")
    print(f"Retry queue size: {metrics['retry_queue_size']}")
    print(f"Uptime: {metrics['uptime_seconds']:.2f} seconds")
    print("====================\n")


def main():
    # Create the message queue
    queue = PriorityMessageQueue(max_size=10000)
    
    # Create producer and handler
    producer = MessageProducer(queue)
    handler = MessageHandler(queue, worker_count=24, max_retries=3)
    
    # Register processors
    handler.register_processor("default", sample_processor)
    handler.register_processor("critical", sample_processor)
    handler.register_processor("batch", sample_processor)
    handler.register_processor("slow", slow_processor)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        handler.stop()
        print_metrics(handler)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the handler
    handler.start()
    
    # Run load test
    load_test(producer, num_messages=5000, concurrency=50)
    
    # Monitor queue processing
    start_time = time.time()
    last_metrics_time = start_time
    
    try:
        while time.time() - start_time < 30:  # Run for 30 seconds
            # Print metrics every 5 seconds
            if time.time() - last_metrics_time > 5:
                print_metrics(handler)
                last_metrics_time = time.time()
            
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        handler.stop()
        print_metrics(handler)


if __name__ == "__main__":
    main()