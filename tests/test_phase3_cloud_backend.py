"""
Comprehensive tests for Phase 3: Cloud Backend Infrastructure
Tests API Gateway, microservices, streaming, and edge-cloud hybrid architecture
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import requests
import redis
from kafka import KafkaProducer, KafkaConsumer
import psycopg2
from sqlalchemy import create_engine, text

# Test configuration
TEST_CONFIG = {
    "api_gateway_url": "http://localhost:8000",
    "user_service_url": "http://localhost:8001", 
    "device_service_url": "http://localhost:8002",
    "redis_host": "localhost",
    "redis_port": 6379,
    "kafka_bootstrap_servers": "localhost:9092",
    "database_url": "postgresql://vigilai:password@localhost:5432/vigilai"
}

class TestCloudBackend:
    """Test suite for cloud backend infrastructure"""
    
    @pytest.fixture
    def setup_services(self):
        """Setup test services"""
        # Initialize connections
        self.redis_client = redis.Redis(
            host=TEST_CONFIG["redis_host"],
            port=TEST_CONFIG["redis_port"],
            decode_responses=True
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=TEST_CONFIG["kafka_bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.kafka_consumer = KafkaConsumer(
            'vigilai.test',
            bootstrap_servers=TEST_CONFIG["kafka_bootstrap_servers"],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.db_engine = create_engine(TEST_CONFIG["database_url"])
        
        yield
        
        # Cleanup
        self.kafka_producer.close()
        self.kafka_consumer.close()
    
    def test_api_gateway_health(self, setup_services):
        """Test API Gateway health endpoint"""
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_api_gateway_rate_limiting(self, setup_services):
        """Test API Gateway rate limiting"""
        # Send multiple requests quickly
        responses = []
        for i in range(10):
            response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
            responses.append(response.status_code)
        
        # Should have some rate limited responses
        assert 429 in responses
    
    def test_user_service_registration(self, setup_services):
        """Test user registration"""
        user_data = {
            "email": "test@vigilai.com",
            "username": "testuser",
            "password": "testpassword123",
            "full_name": "Test User"
        }
        
        response = requests.post(
            f"{TEST_CONFIG['user_service_url']}/auth/register",
            json=user_data
        )
        
        assert response.status_code == 200
        user_response = response.json()
        assert user_response["email"] == user_data["email"]
        assert user_response["username"] == user_data["username"]
    
    def test_user_service_login(self, setup_services):
        """Test user login"""
        # First register a user
        user_data = {
            "email": "login@vigilai.com",
            "username": "loginuser",
            "password": "loginpassword123",
            "full_name": "Login User"
        }
        
        requests.post(
            f"{TEST_CONFIG['user_service_url']}/auth/register",
            json=user_data
        )
        
        # Then login
        login_data = {
            "email": user_data["email"],
            "password": user_data["password"]
        }
        
        response = requests.post(
            f"{TEST_CONFIG['user_service_url']}/auth/login",
            json=login_data
        )
        
        assert response.status_code == 200
        login_response = response.json()
        assert "access_token" in login_response
        assert login_response["token_type"] == "bearer"
    
    def test_device_service_registration(self, setup_services):
        """Test device registration"""
        # First get auth token
        auth_token = self._get_auth_token()
        
        device_data = {
            "device_id": "test-device-001",
            "name": "Test Device",
            "device_type": "vehicle",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "capabilities": {"camera": True, "obd": True}
        }
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = requests.post(
            f"{TEST_CONFIG['device_service_url']}/devices",
            json=device_data,
            headers=headers
        )
        
        assert response.status_code == 200
        device_response = response.json()
        assert device_response["device_id"] == device_data["device_id"]
    
    def test_device_health_update(self, setup_services):
        """Test device health data update"""
        auth_token = self._get_auth_token()
        
        # Register device first
        device_data = {
            "device_id": "health-test-device",
            "name": "Health Test Device",
            "device_type": "vehicle"
        }
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        requests.post(
            f"{TEST_CONFIG['device_service_url']}/devices",
            json=device_data,
            headers=headers
        )
        
        # Update health data
        health_data = {
            "device_id": "health-test-device",
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage": 45.5,
            "memory_usage": 67.2,
            "network_latency": 12.3,
            "is_online": True,
            "error_count": 0
        }
        
        response = requests.post(
            f"{TEST_CONFIG['device_service_url']}/devices/health-test-device/health",
            json=health_data,
            headers=headers
        )
        
        assert response.status_code == 200
    
    def test_kafka_producer_consumer(self, setup_services):
        """Test Kafka producer and consumer"""
        # Send test message
        test_message = {
            "message_id": "test-msg-001",
            "device_id": "test-device",
            "user_id": "test-user",
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "test_message",
            "data": {"test": "data"}
        }
        
        # Produce message
        future = self.kafka_producer.send('vigilai.test', test_message)
        record_metadata = future.get(timeout=10)
        assert record_metadata is not None
        
        # Consume message
        messages = self.kafka_consumer.poll(timeout_ms=5000)
        assert len(messages) > 0
        
        for topic_partition, message_list in messages.items():
            for message in message_list:
                assert message.value["message_id"] == test_message["message_id"]
                break
    
    def test_redis_operations(self, setup_services):
        """Test Redis operations"""
        # Test basic operations
        self.redis_client.set("test_key", "test_value", ex=60)
        value = self.redis_client.get("test_key")
        assert value == "test_value"
        
        # Test hash operations
        self.redis_client.hset("test_hash", "field1", "value1")
        self.redis_client.hset("test_hash", "field2", "value2")
        
        hash_data = self.redis_client.hgetall("test_hash")
        assert hash_data["field1"] == "value1"
        assert hash_data["field2"] == "value2"
    
    def test_database_operations(self, setup_services):
        """Test database operations"""
        with self.db_engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT 1 as test_value")).fetchone()
            assert result.test_value == 1
            
            # Test table creation
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Test insert
            conn.execute(text("""
                INSERT INTO test_table (name) VALUES ('test_record')
            """))
            
            # Test select
            result = conn.execute(text("SELECT * FROM test_table WHERE name = 'test_record'")).fetchone()
            assert result.name == "test_record"
            
            # Cleanup
            conn.execute(text("DROP TABLE test_table"))
    
    def test_edge_gateway_sync(self, setup_services):
        """Test edge gateway synchronization"""
        # Mock edge gateway data
        edge_data = {
            "device_id": "edge-device-001",
            "timestamp": datetime.utcnow().isoformat(),
            "result_type": "fatigue_detection",
            "confidence": 0.85,
            "data": {"fatigue_detected": True, "level": "medium"}
        }
        
        # Test local storage (SQLite)
        import sqlite3
        local_db = sqlite3.connect("test_edge.db")
        cursor = local_db.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                timestamp TEXT,
                result_type TEXT,
                confidence REAL,
                data TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            INSERT INTO processing_results 
            (device_id, timestamp, result_type, confidence, data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            edge_data["device_id"],
            edge_data["timestamp"],
            edge_data["result_type"],
            edge_data["confidence"],
            json.dumps(edge_data["data"])
        ))
        
        local_db.commit()
        
        # Verify data was stored
        cursor.execute("SELECT * FROM processing_results WHERE device_id = ?", (edge_data["device_id"],))
        result = cursor.fetchone()
        assert result is not None
        assert result[2] == edge_data["timestamp"]
        
        # Cleanup
        local_db.close()
        import os
        os.remove("test_edge.db")
    
    def test_microservices_communication(self, setup_services):
        """Test communication between microservices"""
        # Test API Gateway routing to User Service
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/user_service/health")
        assert response.status_code == 200
        
        # Test API Gateway routing to Device Service
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/device_service/health")
        assert response.status_code == 200
    
    def test_data_streaming_pipeline(self, setup_services):
        """Test end-to-end data streaming pipeline"""
        # Simulate data flow: Edge -> Kafka -> Processing -> Storage
        
        # 1. Send data from edge device to Kafka
        edge_message = {
            "message_id": "stream-test-001",
            "device_id": "stream-device",
            "user_id": "stream-user",
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "video_frame",
            "data": {
                "frame_id": "frame-001",
                "landmarks": [{"x": 100, "y": 200}],
                "features": {"eye_aspect_ratio": 0.3}
            }
        }
        
        # Send to Kafka
        future = self.kafka_producer.send('vigilai.video', edge_message)
        record_metadata = future.get(timeout=10)
        assert record_metadata is not None
        
        # 2. Process data (simulate processing)
        processed_data = {
            "device_id": edge_message["device_id"],
            "timestamp": edge_message["timestamp"],
            "result_type": "fatigue_detection",
            "confidence": 0.75,
            "data": {"fatigue_detected": True, "level": "low"}
        }
        
        # 3. Store in database
        with self.db_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO processing_results 
                (device_id, timestamp, result_type, confidence, data)
                VALUES (:device_id, :timestamp, :result_type, :confidence, :data)
            """), processed_data)
        
        # 4. Verify data was stored
        with self.db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM processing_results 
                WHERE device_id = :device_id
            """), {"device_id": processed_data["device_id"]}).fetchone()
            
            assert result is not None
            assert result.confidence == processed_data["confidence"]
    
    def test_system_performance(self, setup_services):
        """Test system performance under load"""
        # Test API Gateway performance
        start_time = time.time()
        
        for i in range(100):
            response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle 100 requests in reasonable time
        assert duration < 10.0  # Less than 10 seconds
        
        # Test Kafka throughput
        start_time = time.time()
        
        for i in range(50):
            message = {
                "message_id": f"perf-test-{i}",
                "device_id": f"device-{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"test": f"data-{i}"}
            }
            future = self.kafka_producer.send('vigilai.test', message)
            future.get(timeout=5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle 50 messages in reasonable time
        assert duration < 5.0  # Less than 5 seconds
    
    def _get_auth_token(self) -> str:
        """Get authentication token for testing"""
        # Register and login a test user
        user_data = {
            "email": "auth@vigilai.com",
            "username": "authuser",
            "password": "authpassword123",
            "full_name": "Auth User"
        }
        
        requests.post(
            f"{TEST_CONFIG['user_service_url']}/auth/register",
            json=user_data
        )
        
        login_data = {
            "email": user_data["email"],
            "password": user_data["password"]
        }
        
        response = requests.post(
            f"{TEST_CONFIG['user_service_url']}/auth/login",
            json=login_data
        )
        
        return response.json()["access_token"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
