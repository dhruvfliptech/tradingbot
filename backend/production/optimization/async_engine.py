"""
Asynchronous Execution Engine
=============================

High-performance async engine for parallel execution of trading operations.
Implements advanced concurrency patterns for maximum throughput.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import uvloop
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import inspect
from asyncio import Queue, PriorityQueue, LifoQueue
import weakref

# Use uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


@dataclass
class AsyncTask(Generic[T]):
    """Async task wrapper with metadata"""
    id: str
    coroutine: Awaitable[T]
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retries: int = 0
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable[[T], None]] = None
    error_handler: Optional[Callable[[Exception], None]] = None
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority.value < other.priority.value


@dataclass
class TaskResult(Generic[T]):
    """Task execution result"""
    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    retries_used: int = 0


class RateLimiter:
    """Token bucket rate limiter for throttling"""
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, returns wait time"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            # Add new tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            return wait_time


class AsyncExecutor:
    """
    Advanced async executor with pooling, batching, and flow control.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Task queues
        self.priority_queue: PriorityQueue = PriorityQueue(
            maxsize=self.config['queue_size']
        )
        self.results: Dict[str, TaskResult] = {}
        
        # Worker pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config['thread_workers'],
            thread_name_prefix='async_thread'
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config['process_workers']
        )
        
        # Semaphores for concurrency control
        self.semaphores = {
            TaskPriority.CRITICAL: asyncio.Semaphore(self.config['critical_concurrency']),
            TaskPriority.HIGH: asyncio.Semaphore(self.config['high_concurrency']),
            TaskPriority.NORMAL: asyncio.Semaphore(self.config['normal_concurrency']),
            TaskPriority.LOW: asyncio.Semaphore(self.config['low_concurrency']),
            TaskPriority.BACKGROUND: asyncio.Semaphore(self.config['background_concurrency'])
        }
        
        # Rate limiters
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Worker tasks
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_execution_time_ms': 0.0,
            'queue_size': 0
        }
        
        logger.info("Async Execution Engine initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'queue_size': 10000,
            'num_workers': 10,
            'thread_workers': 4,
            'process_workers': 2,
            'critical_concurrency': 100,
            'high_concurrency': 50,
            'normal_concurrency': 20,
            'low_concurrency': 10,
            'background_concurrency': 5,
            'default_timeout': 30.0,
            'max_retries': 3,
            'retry_delay': 0.1,
            'batch_size': 100,
            'batch_timeout': 0.01  # 10ms
        }
    
    async def start(self):
        """Start the execution engine"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(self.config['num_workers']):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.workers.append(worker)
        
        # Start batch processor
        asyncio.create_task(self._batch_processor())
        
        logger.info(f"Started {len(self.workers)} async workers")
    
    async def stop(self):
        """Stop the execution engine"""
        self.running = False
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("Async Execution Engine stopped")
    
    async def submit(self, coroutine: Awaitable[T], 
                    priority: TaskPriority = TaskPriority.NORMAL,
                    timeout: Optional[float] = None,
                    retries: int = 0,
                    task_id: Optional[str] = None) -> str:
        """
        Submit a task for execution.
        Returns task ID for tracking.
        """
        task_id = task_id or f"task_{time.time()}_{self.stats['tasks_submitted']}"
        
        task = AsyncTask(
            id=task_id,
            coroutine=coroutine,
            priority=priority,
            timeout=timeout or self.config['default_timeout'],
            retries=retries or self.config['max_retries']
        )
        
        # Add to priority queue
        await self.priority_queue.put((priority.value, task))
        
        self.stats['tasks_submitted'] += 1
        self.stats['queue_size'] = self.priority_queue.qsize()
        
        return task_id
    
    async def submit_batch(self, coroutines: List[Awaitable[T]], 
                          priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """Submit multiple tasks at once"""
        task_ids = []
        
        for coroutine in coroutines:
            task_id = await self.submit(coroutine, priority)
            task_ids.append(task_id)
        
        return task_ids
    
    async def execute_with_timeout(self, coroutine: Awaitable[T], 
                                  timeout: float) -> T:
        """Execute coroutine with timeout"""
        try:
            return await asyncio.wait_for(coroutine, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task timed out after {timeout} seconds")
    
    async def execute_with_retry(self, coroutine_fn: Callable[[], Awaitable[T]], 
                                retries: int = 3,
                                delay: float = 0.1) -> T:
        """Execute with automatic retry on failure"""
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return await coroutine_fn()
            except Exception as e:
                last_error = e
                if attempt < retries:
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    continue
                break
        
        raise last_error
    
    async def map_async(self, func: Callable[[Any], Awaitable[T]], 
                       items: List[Any],
                       concurrency: int = 10) -> List[T]:
        """
        Async map with concurrency control.
        Applies async function to items in parallel.
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_func(item):
            async with semaphore:
                return await func(item)
        
        tasks = [bounded_func(item) for item in items]
        return await asyncio.gather(*tasks)
    
    async def filter_async(self, predicate: Callable[[Any], Awaitable[bool]], 
                          items: List[Any],
                          concurrency: int = 10) -> List[Any]:
        """Async filter with concurrency control"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def check_item(item):
            async with semaphore:
                include = await predicate(item)
                return item if include else None
        
        tasks = [check_item(item) for item in items]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]
    
    async def reduce_async(self, func: Callable[[T, T], Awaitable[T]], 
                          items: List[T],
                          initial: Optional[T] = None) -> T:
        """Async reduce operation"""
        if not items:
            return initial
        
        if initial is None:
            result = items[0]
            items = items[1:]
        else:
            result = initial
        
        for item in items:
            result = await func(result, item)
        
        return result
    
    async def parallel_execute(self, tasks: Dict[str, Awaitable[Any]]) -> Dict[str, Any]:
        """
        Execute multiple named tasks in parallel.
        Returns dict mapping task names to results.
        """
        async def execute_task(name: str, coro: Awaitable[Any]):
            try:
                result = await coro
                return name, result, None
            except Exception as e:
                return name, None, e
        
        # Execute all tasks
        results = await asyncio.gather(*[
            execute_task(name, coro) for name, coro in tasks.items()
        ])
        
        # Process results
        output = {}
        for name, result, error in results:
            if error:
                logger.error(f"Task {name} failed: {error}")
                output[name] = {'error': str(error)}
            else:
                output[name] = result
        
        return output
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run blocking function in thread pool"""
        loop = asyncio.get_event_loop()
        
        if kwargs:
            func = functools.partial(func, **kwargs)
        
        return await loop.run_in_executor(self.thread_pool, func, *args)
    
    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive function in process pool"""
        loop = asyncio.get_event_loop()
        
        if kwargs:
            func = functools.partial(func, **kwargs)
        
        return await loop.run_in_executor(self.process_pool, func, *args)
    
    async def create_rate_limiter(self, name: str, rate: float, capacity: float):
        """Create a named rate limiter"""
        self.rate_limiters[name] = RateLimiter(rate, capacity)
    
    async def rate_limited_execute(self, name: str, coroutine: Awaitable[T]) -> T:
        """Execute with rate limiting"""
        if name not in self.rate_limiters:
            # No rate limiter, execute directly
            return await coroutine
        
        limiter = self.rate_limiters[name]
        wait_time = await limiter.acquire()
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        return await coroutine
    
    async def _worker(self, worker_id: str):
        """Worker task that processes queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                priority, task = await asyncio.wait_for(
                    self.priority_queue.get(), 
                    timeout=1.0
                )
                
                # Process task
                await self._process_task(task)
                
                self.stats['queue_size'] = self.priority_queue.qsize()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: AsyncTask):
        """Process a single task"""
        start_time = time.perf_counter()
        result = TaskResult(task_id=task.id, success=False)
        
        # Get semaphore for priority
        semaphore = self.semaphores[task.priority]
        
        async with semaphore:
            for attempt in range(task.retries + 1):
                try:
                    # Execute with timeout
                    if task.timeout:
                        task_result = await asyncio.wait_for(
                            task.coroutine, 
                            timeout=task.timeout
                        )
                    else:
                        task_result = await task.coroutine
                    
                    # Success
                    result.success = True
                    result.result = task_result
                    result.retries_used = attempt
                    
                    # Call callback if provided
                    if task.callback:
                        task.callback(task_result)
                    
                    break
                    
                except Exception as e:
                    result.error = e
                    
                    if attempt < task.retries:
                        # Retry with exponential backoff
                        await asyncio.sleep(self.config['retry_delay'] * (2 ** attempt))
                        continue
                    
                    # Final failure
                    if task.error_handler:
                        task.error_handler(e)
                    
                    logger.error(f"Task {task.id} failed: {e}")
                    break
        
        # Record execution time
        result.execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Store result
        self.results[task.id] = result
        
        # Update statistics
        if result.success:
            self.stats['tasks_completed'] += 1
        else:
            self.stats['tasks_failed'] += 1
        
        self.stats['avg_execution_time_ms'] = (
            self.stats['avg_execution_time_ms'] * 0.95 + 
            result.execution_time_ms * 0.05
        )
    
    async def _batch_processor(self):
        """Process tasks in batches for efficiency"""
        batch = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Collect batch
                try:
                    priority, task = await asyncio.wait_for(
                        self.priority_queue.get(),
                        timeout=self.config['batch_timeout']
                    )
                    batch.append((priority, task))
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if ready
                should_process = (
                    len(batch) >= self.config['batch_size'] or
                    (time.time() - last_process_time) > self.config['batch_timeout'] and batch
                )
                
                if should_process:
                    # Sort by priority
                    batch.sort(key=lambda x: x[0])
                    
                    # Process batch
                    tasks = [self._process_task(task) for _, task in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    batch.clear()
                    last_process_time = time.time()
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a task to complete"""
        start_time = time.time()
        
        while task_id not in self.results:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.01)
        
        return self.results[task_id]
    
    async def wait_for_all(self, task_ids: List[str], 
                          timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple tasks to complete"""
        tasks = [self.wait_for_task(task_id, timeout) for task_id in task_ids]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'tasks_submitted': self.stats['tasks_submitted'],
            'tasks_completed': self.stats['tasks_completed'],
            'tasks_failed': self.stats['tasks_failed'],
            'success_rate': self.stats['tasks_completed'] / max(1, self.stats['tasks_submitted']),
            'avg_execution_time_ms': round(self.stats['avg_execution_time_ms'], 2),
            'queue_size': self.stats['queue_size'],
            'active_workers': len(self.workers)
        }


class AsyncPipeline:
    """
    Async pipeline for chaining operations.
    """
    
    def __init__(self, executor: AsyncExecutor):
        self.executor = executor
        self.stages: List[Callable] = []
    
    def add_stage(self, func: Callable) -> 'AsyncPipeline':
        """Add a stage to the pipeline"""
        self.stages.append(func)
        return self
    
    async def execute(self, input_data: Any) -> Any:
        """Execute pipeline on input data"""
        result = input_data
        
        for stage in self.stages:
            if asyncio.iscoroutinefunction(stage):
                result = await stage(result)
            else:
                result = await self.executor.run_in_thread(stage, result)
        
        return result
    
    async def execute_parallel(self, inputs: List[Any], 
                              concurrency: int = 10) -> List[Any]:
        """Execute pipeline on multiple inputs in parallel"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_input(input_data):
            async with semaphore:
                return await self.execute(input_data)
        
        tasks = [process_input(inp) for inp in inputs]
        return await asyncio.gather(*tasks)


# Factory function
def create_async_engine(config: Optional[Dict] = None) -> AsyncExecutor:
    """Create and return an async execution engine"""
    return AsyncExecutor(config)