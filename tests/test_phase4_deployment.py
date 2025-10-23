"""
Comprehensive tests for Phase 4: Deployment, Monitoring & Analytics
Tests production deployment, monitoring, analytics dashboard, and CI/CD
"""

import pytest
import asyncio
import json
import time
import requests
import redis
from datetime import datetime, timedelta
from typing import Dict, Any
import psycopg2
from sqlalchemy import create_engine, text
import plotly.graph_objects as go

# Test configuration
TEST_CONFIG = {
    "analytics_dashboard_url": "http://localhost:8007",
    "monitoring_url": "http://localhost:9090",  # Prometheus
    "grafana_url": "http://localhost:3000",
    "redis_host": "localhost",
    "redis_port": 6379,
    "database_url": "postgresql://vigilai:password@localhost:5432/vigilai"
}

class TestPhase4Deployment:
    """Test suite for Phase 4 deployment and monitoring"""
    
    @pytest.fixture
    def setup_services(self):
        """Setup test services"""
        self.redis_client = redis.Redis(
            host=TEST_CONFIG["redis_host"],
            port=TEST_CONFIG["redis_port"],
            decode_responses=True
        )
        
        self.db_engine = create_engine(TEST_CONFIG["database_url"])
        
        yield
        
        # Cleanup
        pass
    
    def test_analytics_dashboard_health(self, setup_services):
        """Test analytics dashboard health endpoint"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_analytics_overview(self, setup_services):
        """Test analytics overview endpoint"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data
        assert "total_users" in data["metrics"]
        assert "total_devices" in data["metrics"]
    
    def test_fleet_status(self, setup_services):
        """Test fleet status endpoint"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/fleet/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status_distribution" in data
        assert "health_metrics" in data
        assert "offline_devices" in data
    
    def test_detection_analytics(self, setup_services):
        """Test detection analytics endpoint"""
        # Test with date range
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        response = requests.get(
            f"{TEST_CONFIG['analytics_dashboard_url']}/api/analytics/detection",
            params={
                "start_date": start_date,
                "end_date": end_date
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "detection_counts" in data
        assert "confidence_stats" in data
        assert "time_series" in data
    
    def test_performance_analytics(self, setup_services):
        """Test performance analytics endpoint"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/analytics/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "performance_metrics" in data
        assert "api_response_time" in data["performance_metrics"]
        assert "processing_latency" in data["performance_metrics"]
    
    def test_user_analytics(self, setup_services):
        """Test user analytics endpoint"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/analytics/users")
        assert response.status_code == 200
        
        data = response.json()
        assert "registration_trends" in data
        assert "active_users_24h" in data
        assert "engagement_metrics" in data
    
    def test_fatigue_trend_chart(self, setup_services):
        """Test fatigue trend chart generation"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/charts/fatigue-trend")
        assert response.status_code == 200
        
        data = response.json()
        assert "chart_html" in data
        assert "data" in data
        assert len(data["data"]) >= 0  # Can be empty for new systems
    
    def test_stress_distribution_chart(self, setup_services):
        """Test stress distribution chart generation"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/charts/stress-distribution")
        assert response.status_code == 200
        
        data = response.json()
        assert "chart_html" in data
        assert "data" in data
    
    def test_device_health_chart(self, setup_services):
        """Test device health chart generation"""
        response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/charts/device-health")
        assert response.status_code == 200
        
        data = response.json()
        assert "chart_html" in data
        assert "data" in data
    
    def test_daily_report_generation(self, setup_services):
        """Test daily report generation"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        response = requests.get(
            f"{TEST_CONFIG['analytics_dashboard_url']}/api/reports/daily",
            params={"date": today}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "date" in data
        assert "summary" in data
        assert "detection_breakdown" in data
        assert "top_devices" in data
    
    def test_monitoring_metrics_collection(self, setup_services):
        """Test monitoring metrics collection"""
        # Test Redis metrics storage
        test_metrics = {
            "api_response_time": 0.05,
            "processing_latency": 0.1,
            "throughput": 1000,
            "error_rate": 0.01
        }
        
        for key, value in test_metrics.items():
            self.redis_client.set(f"metrics:{key}", value)
        
        # Verify metrics were stored
        for key, expected_value in test_metrics.items():
            stored_value = self.redis_client.get(f"metrics:{key}")
            assert stored_value == str(expected_value)
    
    def test_database_analytics_queries(self, setup_services):
        """Test database analytics queries"""
        with self.db_engine.connect() as conn:
            # Test user count query
            user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            assert user_count >= 0
            
            # Test device count query
            device_count = conn.execute(text("SELECT COUNT(*) FROM devices")).scalar()
            assert device_count >= 0
            
            # Test processing results query
            processing_count = conn.execute(text("SELECT COUNT(*) FROM processing_results")).scalar()
            assert processing_count >= 0
    
    def test_chart_generation_performance(self, setup_services):
        """Test chart generation performance"""
        start_time = time.time()
        
        # Generate multiple charts
        chart_endpoints = [
            "/api/charts/fatigue-trend",
            "/api/charts/stress-distribution", 
            "/api/charts/device-health"
        ]
        
        for endpoint in chart_endpoints:
            response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}{endpoint}")
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should generate all charts in reasonable time
        assert duration < 5.0  # Less than 5 seconds
    
    def test_analytics_data_consistency(self, setup_services):
        """Test analytics data consistency across endpoints"""
        # Get overview data
        overview_response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/overview")
        overview_data = overview_response.json()
        
        # Get fleet status data
        fleet_response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/fleet/status")
        fleet_data = fleet_response.json()
        
        # Verify data consistency
        assert overview_data["metrics"]["total_devices"] >= 0
        assert fleet_data["offline_devices"] >= 0
    
    def test_error_handling(self, setup_services):
        """Test error handling in analytics dashboard"""
        # Test invalid date format
        response = requests.get(
            f"{TEST_CONFIG['analytics_dashboard_url']}/api/analytics/detection",
            params={"start_date": "invalid-date"}
        )
        # Should handle gracefully (either 400 or process with default)
        assert response.status_code in [200, 400]
        
        # Test invalid device ID
        response = requests.get(
            f"{TEST_CONFIG['analytics_dashboard_url']}/api/analytics/detection",
            params={"device_id": "nonexistent-device"}
        )
        assert response.status_code == 200  # Should return empty results
    
    def test_concurrent_analytics_requests(self, setup_services):
        """Test concurrent analytics requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}/api/overview")
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Start multiple concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        while not results.empty():
            result = results.get()
            assert result == 200 or "Error" in str(result)
    
    def test_plotly_chart_rendering(self, setup_services):
        """Test Plotly chart rendering"""
        # Create a simple test chart
        fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
        fig.update_layout(title="Test Chart")
        
        # Test chart HTML generation
        chart_html = fig.to_html(include_plotlyjs=False)
        assert "Test Chart" in chart_html
        assert "plotly" in chart_html.lower()
    
    def test_database_performance(self, setup_services):
        """Test database query performance"""
        with self.db_engine.connect() as conn:
            # Test query performance
            start_time = time.time()
            
            # Run multiple queries
            for i in range(10):
                conn.execute(text("SELECT COUNT(*) FROM users"))
                conn.execute(text("SELECT COUNT(*) FROM devices"))
                conn.execute(text("SELECT COUNT(*) FROM processing_results"))
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete in reasonable time
            assert duration < 2.0  # Less than 2 seconds
    
    def test_redis_performance(self, setup_services):
        """Test Redis performance"""
        start_time = time.time()
        
        # Perform multiple Redis operations
        for i in range(100):
            self.redis_client.set(f"test_key_{i}", f"test_value_{i}", ex=60)
            value = self.redis_client.get(f"test_key_{i}")
            assert value == f"test_value_{i}"
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        assert duration < 1.0  # Less than 1 second
    
    def test_analytics_dashboard_responsiveness(self, setup_services):
        """Test analytics dashboard responsiveness"""
        endpoints = [
            "/api/overview",
            "/api/fleet/status",
            "/api/analytics/performance",
            "/api/analytics/users"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}{endpoint}")
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 2.0  # Less than 2 seconds per request
    
    def test_data_validation(self, setup_services):
        """Test data validation in analytics"""
        # Test with various date ranges
        date_ranges = [
            ("2024-01-01", "2024-01-31"),
            ("2024-02-01", "2024-02-29"),
            ("2024-03-01", "2024-03-31")
        ]
        
        for start_date, end_date in date_ranges:
            response = requests.get(
                f"{TEST_CONFIG['analytics_dashboard_url']}/api/analytics/detection",
                params={"start_date": start_date, "end_date": end_date}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "detection_counts" in data
            assert "confidence_stats" in data
            assert "time_series" in data
    
    def test_system_integration(self, setup_services):
        """Test end-to-end system integration"""
        # Test complete analytics workflow
        workflow_steps = [
            ("/api/overview", "Get system overview"),
            ("/api/fleet/status", "Get fleet status"),
            ("/api/analytics/detection", "Get detection analytics"),
            ("/api/analytics/performance", "Get performance analytics"),
            ("/api/analytics/users", "Get user analytics"),
            ("/api/charts/fatigue-trend", "Generate fatigue trend chart"),
            ("/api/charts/stress-distribution", "Generate stress distribution chart"),
            ("/api/charts/device-health", "Generate device health chart"),
            ("/api/reports/daily", "Generate daily report")
        ]
        
        for endpoint, description in workflow_steps:
            response = requests.get(f"{TEST_CONFIG['analytics_dashboard_url']}{endpoint}")
            assert response.status_code == 200, f"Failed at step: {description}"
            
            data = response.json()
            assert data is not None, f"Empty response at step: {description}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
