"""
API Connection Pooling System
=============================

High-performance connection pooling for ultra-low latency API calls.
Implements connection reuse, health checking, and automatic failover.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol
import ssl
import certifi
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration"""
    IDLE = "idle"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ConnectionStats:
    """Connection pool statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    connection_reuse_rate: float = 0.0
    error_rate: float = 0.0
    
    # Connection health
    health_check_success_rate: float = 1.0
    last_health_check: Optional[float] = None


@dataclass
class PooledConnection:
    """Pooled connection wrapper"""
    connection: Any
    endpoint: str
    created_at: float
    last_used: float
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    error_count: int = 0
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def age(self) -> float:
        """Connection age in seconds"""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Idle time in seconds"""
        return time.time() - self.last_used
    
    @property
    def avg_latency(self) -> float:
        """Average latency in milliseconds"""
        if self.latency_history:
            return sum(self.latency_history) / len(self.latency_history)
        return 0.0


class ConnectionPool:
    """
    High-performance connection pool for multiple API endpoints.
    Supports HTTP/HTTPS and WebSocket connections.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Connection pools by endpoint
        self.http_pools: Dict[str, List[PooledConnection]] = {}
        self.ws_pools: Dict[str, List[PooledConnection]] = {}
        
        # HTTP session management
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.connectors: Dict[str, aiohttp.TCPConnector] = {}
        
        # Pool statistics
        self.stats: Dict[str, ConnectionStats] = {}
        
        # Health check tasks
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Lock for thread-safe operations
        self.locks: Dict[str, asyncio.Lock] = {}
        
        logger.info("Connection Pool initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'pool_size': 50,  # Per endpoint
            'max_connections': 100,
            'connection_timeout': 1.0,  # 1 second
            'read_timeout': 5.0,
            'keepalive_timeout': 30,
            'max_keepalive_time': 300,  # 5 minutes
            'health_check_interval': 30,  # seconds
            'retry_attempts': 2,
            'retry_delay': 0.1,
            'enable_compression': True,
            'enable_tcp_nodelay': True,  # Disable Nagle's algorithm
            'enable_tcp_keepalive': True,
            'dns_ttl': 300,  # DNS cache TTL
            'ssl_verify': True
        }
    
    async def initialize(self, endpoints: List[str]):
        """Initialize connection pools for endpoints"""
        for endpoint in endpoints:
            await self._initialize_endpoint(endpoint)
        
        logger.info(f"Initialized connection pools for {len(endpoints)} endpoints")
    
    async def _initialize_endpoint(self, endpoint: str):
        """Initialize pool for specific endpoint"""
        parsed = urlparse(endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Initialize lock
        self.locks[base_url] = asyncio.Lock()
        
        # Initialize stats
        self.stats[base_url] = ConnectionStats()
        
        if parsed.scheme in ('http', 'https'):
            # HTTP/HTTPS pool
            await self._create_http_pool(base_url)
        elif parsed.scheme in ('ws', 'wss'):
            # WebSocket pool
            await self._create_ws_pool(base_url)
        
        # Start health check
        self.health_check_tasks[base_url] = asyncio.create_task(
            self._health_check_worker(base_url)
        )
    
    async def _create_http_pool(self, endpoint: str):
        """Create HTTP connection pool"""
        # Create connector with optimizations
        connector = aiohttp.TCPConnector(
            limit=self.config['pool_size'],
            limit_per_host=self.config['pool_size'],
            ttl_dns_cache=self.config['dns_ttl'],
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=self.config['keepalive_timeout'],
            use_dns_cache=True,
            ssl=self._create_ssl_context() if self.config['ssl_verify'] else False
        )
        
        # Set TCP options for low latency
        if self.config['enable_tcp_nodelay']:
            connector._factory = self._create_tcp_connector
        
        self.connectors[endpoint] = connector
        
        # Create session
        timeout = aiohttp.ClientTimeout(
            total=self.config['read_timeout'],
            connect=self.config['connection_timeout'],
            sock_connect=self.config['connection_timeout'],
            sock_read=self.config['read_timeout']
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Connection': 'keep-alive'},
            compress=self.config['enable_compression']
        )
        
        self.sessions[endpoint] = session
        self.http_pools[endpoint] = []
        
        # Pre-warm connections
        await self._prewarm_http_connections(endpoint)
    
    async def _create_ws_pool(self, endpoint: str):
        """Create WebSocket connection pool"""
        self.ws_pools[endpoint] = []
        
        # Pre-create connections
        for _ in range(min(5, self.config['pool_size'])):
            try:
                conn = await self._create_ws_connection(endpoint)
                if conn:
                    pooled = PooledConnection(
                        connection=conn,
                        endpoint=endpoint,
                        created_at=time.time(),
                        last_used=time.time()
                    )
                    self.ws_pools[endpoint].append(pooled)
            except Exception as e:
                logger.error(f"Failed to create WebSocket connection: {e}")
    
    async def _create_ws_connection(self, endpoint: str) -> Optional[WebSocketClientProtocol]:
        """Create a new WebSocket connection"""
        try:
            ssl_context = self._create_ssl_context() if endpoint.startswith('wss') else None
            
            connection = await websockets.connect(
                endpoint,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024,  # 10MB max message
                compression=None  # Disable compression for speed
            )
            
            return connection
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return None
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create optimized SSL context"""
        context = ssl.create_default_context(cafile=certifi.where())
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Optimize for speed
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        return context
    
    async def _create_tcp_connector(self, *args, **kwargs):
        """Create TCP connection with optimizations"""
        reader, writer = await asyncio.open_connection(*args, **kwargs)
        
        # Set TCP_NODELAY for low latency
        sock = writer.get_extra_info('socket')
        if sock:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Set TCP keepalive
            if self.config['enable_tcp_keepalive']:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        return reader, writer
    
    async def _prewarm_http_connections(self, endpoint: str):
        """Pre-warm HTTP connections"""
        session = self.sessions.get(endpoint)
        if not session:
            return
        
        # Make dummy requests to establish connections
        tasks = []
        for _ in range(min(5, self.config['pool_size'])):
            tasks.append(self._warmup_request(session, endpoint))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats[endpoint].total_connections = self.config['pool_size']
        self.stats[endpoint].idle_connections = self.config['pool_size']
    
    async def _warmup_request(self, session: aiohttp.ClientSession, endpoint: str):
        """Make warmup request"""
        try:
            async with session.head(endpoint, allow_redirects=False) as response:
                pass
        except:
            pass  # Ignore warmup errors
    
    async def request(self, endpoint: str, method: str = 'GET', 
                      **kwargs) -> Tuple[Any, float]:
        """
        Make HTTP request with connection pooling.
        Returns (response, latency_ms).
        """
        parsed = urlparse(endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        session = self.sessions.get(base_url)
        if not session:
            raise ValueError(f"No session for endpoint: {base_url}")
        
        start_time = time.perf_counter()
        
        # Update stats
        stats = self.stats[base_url]
        stats.total_requests += 1
        
        # Retry logic
        last_error = None
        for attempt in range(self.config['retry_attempts']):
            try:
                async with session.request(method, endpoint, **kwargs) as response:
                    data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Update statistics
                    stats.avg_latency_ms = stats.avg_latency_ms * 0.95 + latency_ms * 0.05
                    stats.active_connections = len([c for c in self.connectors[base_url]._acquired])
                    
                    return data, latency_ms
                    
            except Exception as e:
                last_error = e
                if attempt < self.config['retry_attempts'] - 1:
                    await asyncio.sleep(self.config['retry_delay'])
                    continue
        
        # Request failed
        stats.failed_connections += 1
        stats.error_rate = stats.failed_connections / stats.total_requests
        
        raise last_error
    
    async def ws_send_receive(self, endpoint: str, message: Any) -> Tuple[Any, float]:
        """
        Send and receive WebSocket message.
        Returns (response, latency_ms).
        """
        parsed = urlparse(endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        pool = self.ws_pools.get(base_url, [])
        
        # Get available connection
        connection = None
        for pooled in pool:
            if pooled.state == ConnectionState.IDLE:
                connection = pooled
                break
        
        if not connection:
            # Create new connection
            ws = await self._create_ws_connection(endpoint)
            if ws:
                connection = PooledConnection(
                    connection=ws,
                    endpoint=endpoint,
                    created_at=time.time(),
                    last_used=time.time()
                )
                pool.append(connection)
        
        if not connection:
            raise ConnectionError(f"No WebSocket connection available for {endpoint}")
        
        start_time = time.perf_counter()
        
        try:
            # Mark as active
            connection.state = ConnectionState.ACTIVE
            connection.last_used = time.time()
            connection.use_count += 1
            
            # Send message
            await connection.connection.send(message)
            
            # Receive response
            response = await connection.connection.recv()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update stats
            connection.latency_history.append(latency_ms)
            connection.state = ConnectionState.IDLE
            
            stats = self.stats[base_url]
            stats.avg_latency_ms = stats.avg_latency_ms * 0.95 + latency_ms * 0.05
            
            return response, latency_ms
            
        except Exception as e:
            connection.state = ConnectionState.ERROR
            connection.error_count += 1
            raise e
    
    async def _health_check_worker(self, endpoint: str):
        """Background health check worker"""
        while True:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                # HTTP health check
                if endpoint in self.sessions:
                    await self._check_http_health(endpoint)
                
                # WebSocket health check
                if endpoint in self.ws_pools:
                    await self._check_ws_health(endpoint)
                
            except Exception as e:
                logger.error(f"Health check error for {endpoint}: {e}")
    
    async def _check_http_health(self, endpoint: str):
        """Check HTTP connection health"""
        session = self.sessions.get(endpoint)
        if not session:
            return
        
        try:
            async with session.head(endpoint, timeout=2) as response:
                if response.status < 500:
                    stats = self.stats[endpoint]
                    stats.health_check_success_rate = (
                        stats.health_check_success_rate * 0.9 + 0.1
                    )
                    stats.last_health_check = time.time()
        except:
            stats = self.stats[endpoint]
            stats.health_check_success_rate *= 0.9
    
    async def _check_ws_health(self, endpoint: str):
        """Check WebSocket connection health"""
        pool = self.ws_pools.get(endpoint, [])
        
        # Remove dead connections
        healthy_connections = []
        for pooled in pool:
            if pooled.state != ConnectionState.ERROR and pooled.age < self.config['max_keepalive_time']:
                try:
                    # Send ping
                    pong = await pooled.connection.ping()
                    await asyncio.wait_for(pong, timeout=5)
                    healthy_connections.append(pooled)
                except:
                    await pooled.connection.close()
            else:
                if pooled.connection:
                    await pooled.connection.close()
        
        self.ws_pools[endpoint] = healthy_connections
        
        # Replenish pool if needed
        while len(self.ws_pools[endpoint]) < min(5, self.config['pool_size']):
            conn = await self._create_ws_connection(endpoint)
            if conn:
                pooled = PooledConnection(
                    connection=conn,
                    endpoint=endpoint,
                    created_at=time.time(),
                    last_used=time.time()
                )
                self.ws_pools[endpoint].append(pooled)
    
    async def close_endpoint(self, endpoint: str):
        """Close all connections for an endpoint"""
        parsed = urlparse(endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Cancel health check
        if base_url in self.health_check_tasks:
            self.health_check_tasks[base_url].cancel()
        
        # Close HTTP session
        if base_url in self.sessions:
            await self.sessions[base_url].close()
            del self.sessions[base_url]
        
        # Close WebSocket connections
        if base_url in self.ws_pools:
            for pooled in self.ws_pools[base_url]:
                if pooled.connection:
                    await pooled.connection.close()
            del self.ws_pools[base_url]
    
    async def shutdown(self):
        """Shutdown all connection pools"""
        # Cancel all health checks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Close all HTTP sessions
        for session in self.sessions.values():
            await session.close()
        
        # Close all WebSocket connections
        for pool in self.ws_pools.values():
            for pooled in pool:
                if pooled.connection:
                    await pooled.connection.close()
        
        logger.info("Connection Pool shutdown complete")
    
    def get_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if endpoint:
            parsed = urlparse(endpoint)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            stats = self.stats.get(base_url)
            
            if stats:
                return {
                    'endpoint': base_url,
                    'total_requests': stats.total_requests,
                    'avg_latency_ms': round(stats.avg_latency_ms, 2),
                    'error_rate': round(stats.error_rate, 4),
                    'active_connections': stats.active_connections,
                    'health_check_success_rate': round(stats.health_check_success_rate, 4)
                }
        
        # Return all stats
        return {
            endpoint: {
                'total_requests': stats.total_requests,
                'avg_latency_ms': round(stats.avg_latency_ms, 2),
                'error_rate': round(stats.error_rate, 4),
                'active_connections': stats.active_connections,
                'health_check_success_rate': round(stats.health_check_success_rate, 4)
            }
            for endpoint, stats in self.stats.items()
        }


# Factory function
def create_connection_pool(config: Optional[Dict] = None) -> ConnectionPool:
    """Create and return a connection pool instance"""
    return ConnectionPool(config)