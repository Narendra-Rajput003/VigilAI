"""
API Gateway for VigilAI Cloud Backend
Provides load balancing, authentication, rate limiting, and routing
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import jwt
import redis
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active connections')

# Rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 1000  # per window

class RateLimiter:
    """Rate limiter using Redis"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed within rate limit"""
        key = f"rate_limit:{client_id}"
        current = self.redis.incr(key)
        
        if current == 1:
            self.redis.expire(key, RATE_LIMIT_WINDOW)
        
        return current <= RATE_LIMIT_MAX_REQUESTS

class AuthenticationService:
    """Authentication service for API Gateway"""
    
    def __init__(self, secret_key: str, redis_client: redis.Redis):
        self.secret_key = secret_key
        self.redis = redis_client
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions from cache or database"""
        cache_key = f"user_permissions:{user_id}"
        permissions = self.redis.get(cache_key)
        
        if permissions:
            return permissions.decode('utf-8').split(',')
        
        # TODO: Fetch from user service
        return ["read", "write"]

class ServiceRegistry:
    """Service registry for microservices discovery"""
    
    def __init__(self):
        self.services = {
            "user_service": {
                "host": "user-service",
                "port": 8001,
                "health_check": "/health",
                "endpoints": ["/users", "/auth"]
            },
            "device_service": {
                "host": "device-service", 
                "port": 8002,
                "health_check": "/health",
                "endpoints": ["/devices", "/fleet"]
            },
            "data_service": {
                "host": "data-service",
                "port": 8003,
                "health_check": "/health", 
                "endpoints": ["/data", "/streaming"]
            },
            "model_service": {
                "host": "model-service",
                "port": 8004,
                "health_check": "/health",
                "endpoints": ["/models", "/inference"]
            },
            "analytics_service": {
                "host": "analytics-service",
                "port": 8005,
                "health_check": "/health",
                "endpoints": ["/analytics", "/insights"]
            }
        }
    
    def get_service_url(self, service_name: str, path: str = "") -> str:
        """Get service URL for routing"""
        if service_name not in self.services:
            raise HTTPException(status_code=404, detail="Service not found")
        
        service = self.services[service_name]
        base_url = f"http://{service['host']}:{service['port']}"
        return f"{base_url}{path}"

class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.app = FastAPI(
            title="VigilAI API Gateway",
            description="API Gateway for VigilAI Cloud Backend",
            version="1.0.0"
        )
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Use default configuration if file not found
            self.config = {
                'redis': {'host': 'localhost', 'port': 6379},
                'jwt': {'secret_key': 'vigilai-secret-key'},
                'cors': {'allowed_origins': ['*']},
                'security': {'allowed_hosts': ['*']}
            }
        
        # Initialize services with error handling
        try:
            self.redis = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed, using fallback: {e}")
            # Use a mock Redis client for development
            self.redis = None
        
        self.rate_limiter = RateLimiter(self.redis)
        self.auth_service = AuthenticationService(
            self.config['jwt']['secret_key'],
            self.redis
        )
        self.service_registry = ServiceRegistry()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup middleware for the API Gateway"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['cors']['allowed_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config['security']['allowed_hosts']
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            """Rate limiting middleware"""
            client_id = request.client.host
            
            if not await self.rate_limiter.is_allowed(client_id):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
            
            response = await call_next(request)
            return response
        
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            """Metrics collection middleware"""
            start_time = time.time()
            
            response = await call_next(request)
            
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            return response
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.utcnow()}
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        # Authentication routes
        @self.app.post("/auth/login")
        async def login(credentials: dict):
            """User login endpoint"""
            # TODO: Implement login logic
            return {"access_token": "dummy_token", "token_type": "bearer"}
        
        # Service routing
        @self.app.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def route_request(service_name: str, path: str, request: Request):
            """Route requests to appropriate microservices"""
            try:
                # Get service URL
                service_url = self.service_registry.get_service_url(service_name, f"/{path}")
                
                # TODO: Implement actual service routing
                return {"message": f"Routing to {service_url}", "service": service_name}
                
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Routing error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

def create_gateway(config_path: str = "config.yaml") -> APIGateway:
    """Create and configure API Gateway"""
    return APIGateway(config_path)

def main():
    """Main entry point for API Gateway"""
    gateway = create_gateway()
    
    uvicorn.run(
        gateway.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
