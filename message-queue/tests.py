import pytest
import asyncio
import time
import random
import threading
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
from typing import Dict, List, Any

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

# Test fixtures
@pytest.fixture
def queue():
    return PriorityMessageQueue(max_size=100)

@pytest.fixture
def producer(queue):
    return MessageProducer(queue)

@pytest.fixture
def metrics():
    return QueueMetrics()

@pytest.fixture
def handler(queue):
    handler = MessageHandler(queue, worker_count=5, max_retries=2)
    return handler

# Helper async mock processor
async def dummy_processor(message):
    return True

async def failing_processor(message):
    return False

# Message Tests
class TestMessage:
    def test_message_creation(self):
        message = Message(id="test-id", payload={"data": "test"}, timestamp=123.45, priority=1)
        
        assert message.id == "test-id"
        assert message.payload == {"data": "test"}
        assert message.timestamp == 123.45
        assert message.priority == 1
        
    def test_message_string_representation(self):
        message = Message(id="test-id", payload={"data": "test"}, timestamp=123.45, priority=1)
        
        string_rep = str(message)
        assert "test-id" in string_rep
        assert "priority=1" in string_rep
        assert "timestamp=123.45" in string_rep

# Queue Tests
class TestPriorityMessageQueue:
    def test_queue_put_get(self, queue):
        # Create test message
        message = Message(id="test-id", payload={"data": "test"}, timestamp=123.45)
        
        # Put in queue
        assert queue.put(message) == True
        assert queue.size() == 1
        
        # Get from queue
        retrieved = queue.get()
        assert retrieved.id == "test-id"
        assert retrieved.payload == {"data": "test"}
        assert queue.size() == 0
    
    def test_queue_priority_ordering(self, queue):
        # Add messages with different priorities
        high_priority = Message(id="high", payload="high", timestamp=100.0, priority=0)
        medium_priority = Message(id="medium", payload="medium", timestamp=99.0, priority=1)
        low_priority = Message(id="low", payload="low", timestamp=98.0, priority=2)
        
        # Add out of priority order
        queue.put(low_priority)
        queue.put(high_priority)
        queue.put(medium_priority)
        
        # Should retrieve in priority order
        assert queue.get().id == "high"
        assert queue.get().id == "medium"
        assert queue.get().id == "low"
    
    def test_queue_fifo_within_priority(self, queue):
        # Messages with same priority should be FIFO
        msg1 = Message(id="first", payload="first", timestamp=100.0, priority=1)
        msg2 = Message(id="second", payload="second", timestamp=101.0, priority=1)
        
        queue.put(msg1)
        queue.put(msg2)
        
        assert queue.get().id == "first"
        assert queue.get().id == "second"
    
    def test_queue_max_size(self):
        small_queue = PriorityMessageQueue(max_size=2)
        
        msg1 = Message(id="1", payload="1", timestamp=1.0)
        msg2 = Message(id="2", payload="2", timestamp=2.0)
        msg3 = Message(id="3", payload="3", timestamp=3.0)
        
        assert small_queue.put(msg1) == True
        assert small_queue.put(msg2) == True
        assert small_queue.put(msg3) == False  # Should fail, queue is full
        
        assert small_queue.size() == 2
    
    def test_queue_empty_get(self, queue):
        assert queue.get(block=False) is None
        
    def test_queue_timeout(self, queue):
        assert queue.get(block=True, timeout=0.1) is None

# Metrics Tests
class TestQueueMetrics:
    def test_metrics_recording(self, metrics):
        # Record some processed messages
        metrics.record_processed(0.1)
        metrics.record_processed(0.2)
        metrics.record_processed(0.3)
        
        # Record a failure
        metrics.record_failed()
        
        # Check metrics
        current_metrics = metrics.get_metrics()
        assert current_metrics["processed_messages"] == 3
        assert current_metrics["failed_messages"] == 1
        assert current_metrics["avg_processing_time"] == 0.2  # (0.1 + 0.2 + 0.3) / 3
        
    def test_metrics_uptime(self, metrics):
        # Get initial metrics
        time.sleep(0.1)
        metrics_data = metrics.get_metrics()
        
        # Uptime should be greater than 0.1
        assert metrics_data["uptime_seconds"] >= 0.1
        
    def test_messages_per_second(self, metrics):
        start_time = time.time()
        metrics.start_time = start_time
        
        # Add 10 messages over 0.1 seconds
        for _ in range(10):
            metrics.record_processed(0.01)
            
        # Force uptime to be exactly 0.1 for testing
        with patch.object(time, 'time', return_value=start_time + 0.1):
            metrics_data = metrics.get_metrics()
            assert metrics_data["messages_per_second"] == 100.0  # 10 msgs / 0.1 seconds

# Producer Tests
class TestMessageProducer:
    def test_send_message(self, producer, queue):
        message_id = producer.send_message({"data": "test"}, priority=1)
        
        # Verify ID is returned
        assert isinstance(message_id, str)
        assert len(message_id) > 0
        
        # Verify message is in queue
        assert queue.size() == 1
        message = queue.get()
        assert message.id == message_id
        assert message.payload == {"data": "test"}
        assert message.priority == 1
    
    def test_send_batch(self, producer, queue):
        payloads = [{"data": f"test-{i}"} for i in range(5)]
        
        message_ids = producer.send_batch(payloads, priority=2)
        
        # Verify IDs returned
        assert len(message_ids) == 5
        assert all(isinstance(id, str) for id in message_ids)
        
        # Verify messages in queue
        assert queue.size() == 5
        
        # Get all messages and check them
        messages = [queue.get() for _ in range(5)]
        for i, message in enumerate(messages):
            assert message.id == message_ids[i]
            assert message.payload == {"data": f"test-{i}"}
            assert message.priority == 2

# Handler Tests
class TestMessageHandler:
    def test_register_processor(self, handler):
        # Register processors
        handler.register_processor("test_type", dummy_processor)
        
        # Check if processor is registered
        assert "test_type" in handler.processors
        assert handler.processors["test_type"] == dummy_processor
    
    @pytest.mark.asyncio
    async def test_process_message(self, handler):
        # Register processor
        mock_processor = AsyncMock(return_value=True)
        handler.register_processor("test_type", mock_processor)
        
        # Create message
        message = Message(
            id="test-id", 
            payload={"type": "test_type", "data": "test"}, 
            timestamp=time.time()
        )
        
        # Process message
        result = await handler.process_message(message)
        
        # Verify processor was called
        assert result == True
        mock_processor.assert_called_once()
        mock_processor.assert_called_with(message)
    
    @pytest.mark.asyncio
    async def test_process_message_no_processor(self, handler):
        # Create message with unknown type
        message = Message(
            id="test-id", 
            payload={"type": "unknown", "data": "test"}, 
            timestamp=time.time()
        )
        
        # Process message
        result = await handler.process_message(message)
        
        # Should fail without a processor
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process_message_exception(self, handler):
        # Register processor that raises exception
        async def error_processor(message):
            raise Exception("Test error")
            
        handler.register_processor("error_type", error_processor)
        
        # Create message
        message = Message(
            id="test-id", 
            payload={"type": "error_type", "data": "test"}, 
            timestamp=time.time()
        )
        
        # Process message
        result = await handler.process_message(message)
        
        # Should handle the exception and return False
        assert result == False
    
    def test_start_stop(self, handler):
        # Start handler
        handler.start()
        
        # Verify running state
        assert handler.running == True
        assert len(handler.workers) == 5  # Default worker count
        
        # Stop handler
        handler.stop()
        
        # Verify stopped state
        assert handler.running == False
        assert len(handler.workers) == 0
    
    def test_get_metrics(self, handler):
        # Get metrics
        metrics = handler.get_metrics()
        
        # Verify metrics structure
        assert "processed_messages" in metrics
        assert "failed_messages" in metrics
        assert "avg_processing_time" in metrics
        assert "messages_per_second" in metrics
        assert "uptime_seconds" in metrics
        assert "queue_size" in metrics
        assert "retry_queue_size" in metrics
        assert "worker_count" in metrics
        
        # Initially all counts should be 0
        assert metrics["processed_messages"] == 0
        assert metrics["failed_messages"] == 0

# Integration Tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_message_flow(self, queue, handler):
        # Register processor
        processor_called = asyncio.Event()
        processed_message_id = None
        
        async def test_processor(message):
            nonlocal processed_message_id
            processed_message_id = message.id
            processor_called.set()
            return True
        
        handler.register_processor("default", test_processor)
        handler.start()
        
        try:
            # Send message
            producer = MessageProducer(queue)
            message_id = producer.send_message({"data": "test_integration"})
            
            # Wait for processor to be called (with timeout)
            await asyncio.wait_for(processor_called.wait(), timeout=2.0)
            
            # Verify message was processed
            assert processed_message_id == message_id
            
            # Allow time for metrics to update
            await asyncio.sleep(0.1)
            
            # Check metrics
            metrics = handler.get_metrics()
            assert metrics["processed_messages"] >= 1
        finally:
            handler.stop()
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, queue):
        # Create handler with retry capabilities
        handler = MessageHandler(queue, worker_count=2, max_retries=2)
        
        # Track processor calls
        call_count = 0
        
        async def failing_then_succeeding(message):
            nonlocal call_count
            call_count += 1
            
            # Fail first time, succeed second time
            return call_count > 1
        
        handler.register_processor("default", failing_then_succeeding)
        handler.start()
        
        try:
            # Send message
            producer = MessageProducer(queue)
            producer.send_message({"data": "retry_test"})
            
            # Wait for processing to complete
            await asyncio.sleep(1.0)
            
            # Should have been called at least twice due to retry
            assert call_count >= 2
            
            # Check metrics
            metrics = handler.get_metrics()
            assert metrics["processed_messages"] >= 1  # Eventually succeeded
            assert metrics["failed_messages"] >= 1  # Failed on first attempt
        finally:
            handler.stop()
    
    def test_concurrency(self, queue):
        # Test that handler can process multiple messages concurrently
        handler = MessageHandler(queue, worker_count=10)
        
        # Create a processor that takes some time
        process_times = []
        completion_order = []
        lock = threading.Lock()
        
        async def slow_processor(message):
            # Get message sequence from payload
            seq = message.payload.get("sequence")
            
            # Random processing time between 0.1-0.3 seconds
            sleep_time = random.uniform(0.1, 0.3)
            start_time = time.time()
            await asyncio.sleep(sleep_time)
            end_time = time.time()
            
            # Record processing time and completion order
            with lock:
                process_times.append(end_time - start_time)
                completion_order.append(seq)
            
            return True
        
        handler.register_processor("default", slow_processor)
        handler.start()
        
        try:
            # Send multiple messages
            producer = MessageProducer(queue)
            for i in range(20):
                producer.send_message({"sequence": i})
            
            # Wait for all to be processed
            time.sleep(1.0)
            
            # Verify all messages were processed
            metrics = handler.get_metrics()
            assert metrics["processed_messages"] == 20
            
            # Verify processing happened concurrently by checking that completion
            # order differs from submission order
            assert completion_order != list(range(20))
            
            # The total time should be less than the sum of individual times
            # which indicates concurrent processing
            total_time = max(process_times) if process_times else 0
            sum_times = sum(process_times)
            assert total_time < sum_times
        finally:
            handler.stop()

    def test_load(self, queue):
        """Test handling a large number of messages"""
        handler = MessageHandler(queue, worker_count=20)
        
        # Simple fast processor
        async def fast_processor(message):
            return True
        
        handler.register_processor("default", fast_processor)
        handler.start()
        
        try:
            # Send a large batch of messages
            producer = MessageProducer(queue)
            message_count = 1000
            
            start_time = time.time()
            for i in range(message_count):
                producer.send_message({"index": i})
            
            # Wait for processing to complete
            max_wait = 10.0  # Maximum wait time in seconds
            wait_start = time.time()
            
            while True:
                metrics = handler.get_metrics()
                if metrics["processed_messages"] >= message_count:
                    break
                    
                if time.time() - wait_start > max_wait:
                    pytest.fail(f"Timed out waiting for messages to be processed. "
                               f"Processed {metrics['processed_messages']}/{message_count}")
                    break
                    
                time.sleep(0.1)
            
            end_time = time.time()
            
            # Calculate throughput
            throughput = message_count / (end_time - start_time)
            
            # Verify all messages were processed
            metrics = handler.get_metrics()
            assert metrics["processed_messages"] >= message_count
            
            # Log throughput for information (not a strict test)
            print(f"\nThroughput: {throughput:.2f} messages/second")
            
            # Should be reasonably fast (adjust threshold based on your system)
            assert throughput > 100, "Throughput is lower than expected"
            
        finally:
            handler.stop()

    def test_priority_handling(self, queue):
        """Test that higher priority messages are processed first"""
        handler = MessageHandler(queue, worker_count=1)  # Single worker to test ordering
        
        processed_ids = []
        
        async def tracking_processor(message):
            processed_ids.append(message.id)
            await asyncio.sleep(0.01)  # Small delay
            return True
        
        handler.register_processor("default", tracking_processor)
        handler.start()
        
        try:
            # Send messages with different priorities
            producer = MessageProducer(queue)
            
            # Low priority (2) messages
            low_ids = []
            for i in range(5):
                msg_id = producer.send_message({"data": f"low-{i}"}, priority=2)
                low_ids.append(msg_id)
            
            # High priority (0) messages
            high_ids = []
            for i in range(5):
                msg_id = producer.send_message({"data": f"high-{i}"}, priority=0)
                high_ids.append(msg_id)
            
            # Medium priority (1) messages
            medium_ids = []
            for i in range(5):
                msg_id = producer.send_message({"data": f"medium-{i}"}, priority=1)
                medium_ids.append(msg_id)
            
            # Wait for processing
            time.sleep(1.0)
            
            # Verify all messages were processed
            assert len(processed_ids) == 15
            
            # Check if high priority messages were processed before lower priority ones
            # Find positions of each priority group
            high_positions = [processed_ids.index(id) for id in high_ids]
            medium_positions = [processed_ids.index(id) for id in medium_ids]
            low_positions = [processed_ids.index(id) for id in low_ids]
            
            # The max position of higher priority should be less than min position of lower priority
            assert max(high_positions) < min(medium_positions)
            assert max(medium_positions) < min(low_positions)
            
        finally:
            handler.stop()