"""
User Service for VigilAI Cloud Backend
Handles user management, authentication, and authorization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "postgresql://vigilai:password@localhost:5432/vigilai"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Security
security = HTTPBearer()
JWT_SECRET = "vigilai-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    preferences = Column(Text)  # JSON string for user preferences
    role = Column(String, default="user")  # user, admin, fleet_manager

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, index=True)
    token = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str
    preferences: Optional[Dict[str, Any]] = {}

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    role: str
    preferences: Dict[str, Any]

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class UserService:
    """User service implementation"""
    
    def __init__(self):
        self.app = FastAPI(title="User Service", version="1.0.0")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/auth/register", response_model=UserResponse)
        async def register_user(user_data: UserCreate):
            """Register a new user"""
            db = SessionLocal()
            try:
                # Check if user already exists
                existing_user = db.query(User).filter(
                    (User.email == user_data.email) | 
                    (User.username == user_data.username)
                ).first()
                
                if existing_user:
                    raise HTTPException(
                        status_code=400,
                        detail="User with this email or username already exists"
                    )
                
                # Hash password
                hashed_password = bcrypt.hashpw(
                    user_data.password.encode('utf-8'), 
                    bcrypt.gensalt()
                ).decode('utf-8')
                
                # Create user
                user = User(
                    email=user_data.email,
                    username=user_data.username,
                    hashed_password=hashed_password,
                    full_name=user_data.full_name,
                    preferences=user_data.preferences
                )
                
                db.add(user)
                db.commit()
                db.refresh(user)
                
                return UserResponse(
                    id=user.id,
                    email=user.email,
                    username=user.username,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    is_verified=user.is_verified,
                    created_at=user.created_at,
                    role=user.role,
                    preferences=user.preferences or {}
                )
                
            except Exception as e:
                db.rollback()
                logger.error(f"Registration error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
            finally:
                db.close()
        
        @self.app.post("/auth/login", response_model=LoginResponse)
        async def login(login_data: LoginRequest):
            """User login"""
            db = SessionLocal()
            try:
                # Find user
                user = db.query(User).filter(User.email == login_data.email).first()
                
                if not user or not user.is_active:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid credentials"
                    )
                
                # Verify password
                if not bcrypt.checkpw(
                    login_data.password.encode('utf-8'),
                    user.hashed_password.encode('utf-8')
                ):
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid credentials"
                    )
                
                # Generate JWT token
                token_data = {
                    "user_id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "exp": datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION)
                }
                
                token = jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
                
                # Store session
                session = UserSession(
                    user_id=user.id,
                    token=token,
                    expires_at=datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION)
                )
                db.add(session)
                db.commit()
                
                # Cache user data in Redis
                redis_client.setex(
                    f"user:{user.id}",
                    JWT_EXPIRATION,
                    str(user.id)
                )
                
                return LoginResponse(
                    access_token=token,
                    token_type="bearer",
                    expires_in=JWT_EXPIRATION,
                    user=UserResponse(
                        id=user.id,
                        email=user.email,
                        username=user.username,
                        full_name=user.full_name,
                        is_active=user.is_active,
                        is_verified=user.is_verified,
                        created_at=user.created_at,
                        role=user.role,
                        preferences=user.preferences or {}
                    )
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Login error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
            finally:
                db.close()
        
        @self.app.get("/users/me", response_model=UserResponse)
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Get current user information"""
            try:
                # Verify token
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Get user from database
                db = SessionLocal()
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                return UserResponse(
                    id=user.id,
                    email=user.email,
                    username=user.username,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    is_verified=user.is_verified,
                    created_at=user.created_at,
                    role=user.role,
                    preferences=user.preferences or {}
                )
                
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail="Token expired")
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
            except Exception as e:
                logger.error(f"Get user error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.put("/users/me", response_model=UserResponse)
        async def update_current_user(
            user_update: UserUpdate,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Update current user information"""
            try:
                # Verify token
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Update user
                db = SessionLocal()
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                if user_update.full_name is not None:
                    user.full_name = user_update.full_name
                
                if user_update.preferences is not None:
                    user.preferences = user_update.preferences
                
                user.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(user)
                
                return UserResponse(
                    id=user.id,
                    email=user.email,
                    username=user.username,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    is_verified=user.is_verified,
                    created_at=user.created_at,
                    role=user.role,
                    preferences=user.preferences or {}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Update user error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/auth/logout")
        async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """User logout"""
            try:
                # Verify token
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Deactivate session
                db = SessionLocal()
                session = db.query(UserSession).filter(
                    UserSession.user_id == user_id,
                    UserSession.token == credentials.credentials
                ).first()
                
                if session:
                    session.is_active = False
                    db.commit()
                
                # Remove from Redis cache
                redis_client.delete(f"user:{user_id}")
                
                return {"message": "Successfully logged out"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Logout error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "user_service"}

def create_user_service() -> UserService:
    """Create user service instance"""
    return UserService()

def main():
    """Main entry point for user service"""
    service = create_user_service()
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

if __name__ == "__main__":
    main()
