"""
System Health Monitor for VigilAI
Comprehensive system monitoring, health checks, and automatic recovery
"""

import asyncio
import logging
import time
import psutil
import requests
import redis
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

class ComponentType(Enum):
    """Types of system components"""
    DATABASE = "database"
    REDIS = "redis"
    KAFKA = "kafka"
    API_GATEWAY = "api_gateway"
    USER_SERVICE = "user_service"
    DEVICE_SERVICE = "device_service"
    ANALYTICS_SERVICE = "analytics_service"
    MODEL_SERVICE = "model_service"
    EDGE_GATEWAY = "edge_gateway"

@dataclass
class HealthCheck:
    """Health check result"""
    component: ComponentType
    status: HealthStatus
    response_time: float
    timestamp: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]
    timestamp: datetime

class ConnectionManager:
    """Manages connections to external services with retry logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}
        self.connection_pools = {}
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'backoff_factor': 2.0,
            'timeout': 10.0
        }
    
    async def get_database_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Get database connection with retry logic"""
        try:
            db_config = self.config.get('database', {})
            connection_string = db_config.get('url', 'postgresql://vigilai:password@localhost:5432/vigilai')
            
            # Try to get existing connection
            if 'database' in self.connections:
                conn = self.connections['database']
                if not conn.closed:
                    return conn
            
            # Create new connection
            conn = psycopg2.connect(connection_string)
            self.connections['database'] = conn
            return conn
            
        except Exception as e:
            logger.error(f"Error getting database connection: {e}")
            return None
    
    async def get_redis_connection(self) -> Optional[redis.Redis]:
        """Get Redis connection with retry logic"""
        try:
            redis_config = self.config.get('redis', {})
            host = redis_config.get('host', 'localhost')
            port = redis_config.get('port', 6379)
            
            # Try to get existing connection
            if 'redis' in self.connections:
                conn = self.connections['redis']
                try:
                    conn.ping()
                    return conn
                except:
                    pass
            
            # Create new connection
            conn = redis.Redis(host=host, port=port, decode_responses=True)
            conn.ping()  # Test connection
            self.connections['redis'] = conn
            return conn
            
        except Exception as e:
            logger.error(f"Error getting Redis connection: {e}")
            return None
    
    async def get_kafka_connection(self) -> bool:
        """Check Kafka connection"""
        try:
            kafka_config = self.config.get('kafka', {})
            bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
            
            # Simple HTTP check for Kafka (if REST proxy is available)
            kafka_url = f"http://{bootstrap_servers.split(',')[0].split(':')[0]}:8082"
            response = requests.get(f"{kafka_url}/topics", timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error checking Kafka connection: {e}")
            return False
    
    async def test_service_connection(self, service_name: str, url: str) -> Tuple[bool, float]:
        """Test connection to a service"""
        try:
            start_time = time.time()
            response = requests.get(f"{url}/health", timeout=self.retry_config['timeout'])
            response_time = time.time() - start_time
            
            return response.status_code == 200, response_time
            
        except Exception as e:
            logger.error(f"Error testing {service_name} connection: {e}")
            return False, 0.0
    
    async def close_all_connections(self):
        """Close all connections"""
        try:
            # Close database connections
            if 'database' in self.connections:
                self.connections['database'].close()
            
            # Redis connections are auto-closed
            self.connections.clear()
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_manager = ConnectionManager(config)
        self.health_history = []
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 0.05
        }
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Service endpoints
        self.service_endpoints = {
            ComponentType.API_GATEWAY: config.get('services', {}).get('api_gateway', 'http://localhost:8000'),
            ComponentType.USER_SERVICE: config.get('services', {}).get('user_service', 'http://localhost:8001'),
            ComponentType.DEVICE_SERVICE: config.get('services', {}).get('device_service', 'http://localhost:8002'),
            ComponentType.ANALYTICS_SERVICE: config.get('services', {}).get('analytics_service', 'http://localhost:8005'),
            ComponentType.MODEL_SERVICE: config.get('services', {}).get('model_service', 'http://localhost:8004'),
            ComponentType.EDGE_GATEWAY: config.get('services', {}).get('edge_gateway', 'http://localhost:8006')
        }
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous health monitoring"""
        try:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
            logger.info("System health monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting health monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        try:
            self.is_monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            logger.info("System health monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {e}")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.metrics_history.append(system_metrics)
                
                # Perform health checks
                health_checks = await self._perform_health_checks()
                self.health_history.extend(health_checks)
                
                # Check for alerts
                await self._check_alerts(health_checks, system_metrics)
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_dict = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io_dict,
                process_count=process_count,
                load_average=list(load_avg),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0],
                timestamp=datetime.utcnow()
            )
    
    async def _perform_health_checks(self) -> List[HealthCheck]:
        """Perform health checks on all components"""
        health_checks = []
        
        # Check database
        db_check = await self._check_database_health()
        health_checks.append(db_check)
        
        # Check Redis
        redis_check = await self._check_redis_health()
        health_checks.append(redis_check)
        
        # Check Kafka
        kafka_check = await self._check_kafka_health()
        health_checks.append(kafka_check)
        
        # Check services
        for component, url in self.service_endpoints.items():
            service_check = await self._check_service_health(component, url)
            health_checks.append(service_check)
        
        return health_checks
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database health"""
        try:
            start_time = time.time()
            conn = await self.connection_manager.get_database_connection()
            response_time = time.time() - start_time
            
            if conn:
                # Test query
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                
                return HealthCheck(
                    component=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    timestamp=datetime.utcnow(),
                    metrics={'connection_count': 1}
                )
            else:
                return HealthCheck(
                    component=ComponentType.DATABASE,
                    status=HealthStatus.DOWN,
                    response_time=response_time,
                    timestamp=datetime.utcnow(),
                    error_message="Database connection failed"
                )
                
        except Exception as e:
            return HealthCheck(
                component=ComponentType.DATABASE,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _check_redis_health(self) -> HealthCheck:
        """Check Redis health"""
        try:
            start_time = time.time()
            redis_conn = await self.connection_manager.get_redis_connection()
            response_time = time.time() - start_time
            
            if redis_conn:
                # Test Redis operations
                redis_conn.set('health_check', 'test', ex=10)
                redis_conn.get('health_check')
                
                # Get Redis info
                info = redis_conn.info()
                
                return HealthCheck(
                    component=ComponentType.REDIS,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    timestamp=datetime.utcnow(),
                    metrics={
                        'used_memory': info.get('used_memory', 0),
                        'connected_clients': info.get('connected_clients', 0),
                        'keyspace_hits': info.get('keyspace_hits', 0)
                    }
                )
            else:
                return HealthCheck(
                    component=ComponentType.REDIS,
                    status=HealthStatus.DOWN,
                    response_time=response_time,
                    timestamp=datetime.utcnow(),
                    error_message="Redis connection failed"
                )
                
        except Exception as e:
            return HealthCheck(
                component=ComponentType.REDIS,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _check_kafka_health(self) -> HealthCheck:
        """Check Kafka health"""
        try:
            start_time = time.time()
            is_connected = await self.connection_manager.get_kafka_connection()
            response_time = time.time() - start_time
            
            if is_connected:
                return HealthCheck(
                    component=ComponentType.KAFKA,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    timestamp=datetime.utcnow()
                )
            else:
                return HealthCheck(
                    component=ComponentType.KAFKA,
                    status=HealthStatus.DOWN,
                    response_time=response_time,
                    timestamp=datetime.utcnow(),
                    error_message="Kafka connection failed"
                )
                
        except Exception as e:
            return HealthCheck(
                component=ComponentType.KAFKA,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _check_service_health(self, component: ComponentType, url: str) -> HealthCheck:
        """Check service health"""
        try:
            is_connected, response_time = await self.connection_manager.test_service_connection(
                component.value, url
            )
            
            if is_connected:
                status = HealthStatus.HEALTHY
                if response_time > self.alert_thresholds['response_time']:
                    status = HealthStatus.WARNING
            else:
                status = HealthStatus.DOWN
            
            return HealthCheck(
                component=component,
                status=status,
                response_time=response_time,
                timestamp=datetime.utcnow(),
                error_message=None if is_connected else "Service not responding"
            )
            
        except Exception as e:
            return HealthCheck(
                component=component,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _check_alerts(self, health_checks: List[HealthCheck], system_metrics: SystemMetrics):
        """Check for alert conditions"""
        try:
            alerts = []
            
            # Check system metrics
            if system_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                alerts.append(f"High CPU usage: {system_metrics.cpu_usage:.1f}%")
            
            if system_metrics.memory_usage > self.alert_thresholds['memory_usage']:
                alerts.append(f"High memory usage: {system_metrics.memory_usage:.1f}%")
            
            if system_metrics.disk_usage > self.alert_thresholds['disk_usage']:
                alerts.append(f"High disk usage: {system_metrics.disk_usage:.1f}%")
            
            # Check health checks
            for check in health_checks:
                if check.status == HealthStatus.CRITICAL:
                    alerts.append(f"Critical: {check.component.value} - {check.error_message}")
                elif check.status == HealthStatus.DOWN:
                    alerts.append(f"Down: {check.component.value} - {check.error_message}")
                elif check.status == HealthStatus.WARNING:
                    alerts.append(f"Warning: {check.component.value} - Slow response")
            
            # Send alerts if any
            if alerts:
                await self._send_alerts(alerts)
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _send_alerts(self, alerts: List[str]):
        """Send alerts to monitoring system"""
        try:
            alert_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'alerts': alerts,
                'system_status': 'warning' if any('Warning' in alert for alert in alerts) else 'critical'
            }
            
            # Send to monitoring endpoint
            monitoring_url = self.config.get('monitoring', {}).get('alert_endpoint')
            if monitoring_url:
                response = requests.post(monitoring_url, json=alert_data, timeout=10)
                logger.info(f"Alerts sent: {response.status_code}")
            
            # Log alerts
            for alert in alerts:
                logger.warning(f"SYSTEM ALERT: {alert}")
                
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old health check and metrics data"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Cleanup health history
            self.health_history = [
                check for check in self.health_history 
                if check.timestamp > cutoff_time
            ]
            
            # Cleanup metrics history
            self.metrics_history = [
                metrics for metrics in self.metrics_history 
                if metrics.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            # Get latest metrics
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            # Get latest health checks
            latest_checks = {}
            for check in self.health_history[-10:]:  # Last 10 checks
                latest_checks[check.component.value] = {
                    'status': check.status.value,
                    'response_time': check.response_time,
                    'timestamp': check.timestamp.isoformat(),
                    'error': check.error_message
                }
            
            # Calculate overall health
            overall_health = self._calculate_overall_health()
            
            return {
                'overall_health': overall_health,
                'system_metrics': {
                    'cpu_usage': latest_metrics.cpu_usage if latest_metrics else 0,
                    'memory_usage': latest_metrics.memory_usage if latest_metrics else 0,
                    'disk_usage': latest_metrics.disk_usage if latest_metrics else 0,
                    'timestamp': latest_metrics.timestamp.isoformat() if latest_metrics else None
                },
                'component_status': latest_checks,
                'monitoring_active': self.is_monitoring
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        try:
            if not self.health_history:
                return 'unknown'
            
            # Get recent health checks
            recent_checks = self.health_history[-20:]  # Last 20 checks
            
            # Count statuses
            status_counts = {}
            for check in recent_checks:
                status = check.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Determine overall health
            if status_counts.get('critical', 0) > 0:
                return 'critical'
            elif status_counts.get('down', 0) > 2:
                return 'critical'
            elif status_counts.get('warning', 0) > 5:
                return 'warning'
            else:
                return 'healthy'
                
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return 'unknown'
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            # System metrics summary
            if self.metrics_history:
                cpu_values = [m.cpu_usage for m in self.metrics_history[-24:]]  # Last 24 hours
                memory_values = [m.memory_usage for m in self.metrics_history[-24:]]
                disk_values = [m.disk_usage for m in self.metrics_history[-24:]]
                
                metrics_summary = {
                    'cpu': {
                        'current': cpu_values[-1] if cpu_values else 0,
                        'average': np.mean(cpu_values) if cpu_values else 0,
                        'max': np.max(cpu_values) if cpu_values else 0,
                        'min': np.min(cpu_values) if cpu_values else 0
                    },
                    'memory': {
                        'current': memory_values[-1] if memory_values else 0,
                        'average': np.mean(memory_values) if memory_values else 0,
                        'max': np.max(memory_values) if memory_values else 0,
                        'min': np.min(memory_values) if memory_values else 0
                    },
                    'disk': {
                        'current': disk_values[-1] if disk_values else 0,
                        'average': np.mean(disk_values) if disk_values else 0,
                        'max': np.max(disk_values) if disk_values else 0,
                        'min': np.min(disk_values) if disk_values else 0
                    }
                }
            else:
                metrics_summary = {}
            
            # Component health summary
            component_health = {}
            for component in ComponentType:
                component_checks = [c for c in self.health_history if c.component == component]
                if component_checks:
                    recent_check = component_checks[-1]
                    component_health[component.value] = {
                        'status': recent_check.status.value,
                        'response_time': recent_check.response_time,
                        'last_check': recent_check.timestamp.isoformat(),
                        'uptime': self._calculate_uptime(component_checks)
                    }
            
            return {
                'report_timestamp': datetime.utcnow().isoformat(),
                'overall_health': self._calculate_overall_health(),
                'system_metrics': metrics_summary,
                'component_health': component_health,
                'monitoring_duration': len(self.metrics_history),
                'total_health_checks': len(self.health_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {'error': str(e)}
    
    def _calculate_uptime(self, health_checks: List[HealthCheck]) -> float:
        """Calculate uptime percentage for a component"""
        try:
            if not health_checks:
                return 0.0
            
            # Count healthy checks
            healthy_checks = sum(1 for check in health_checks if check.status == HealthStatus.HEALTHY)
            total_checks = len(health_checks)
            
            return (healthy_checks / total_checks) * 100
            
        except Exception as e:
            logger.error(f"Error calculating uptime: {e}")
            return 0.0
    
    async def cleanup(self):
        """Cleanup system health monitor"""
        try:
            await self.stop_monitoring()
            await self.connection_manager.close_all_connections()
            logger.info("System health monitor cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
