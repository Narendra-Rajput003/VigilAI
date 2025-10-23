"""
Analytics Dashboard for VigilAI
Real-time analytics, insights, and fleet management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import redis
import psycopg2
from sqlalchemy import create_engine, text
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "postgresql://vigilai:password@localhost:5432/vigilai"
engine = create_engine(DATABASE_URL)

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Security
JWT_SECRET = "vigilai-secret-key"
JWT_ALGORITHM = "HS256"

class AnalyticsDashboard:
    """Analytics dashboard for VigilAI"""
    
    def __init__(self):
        self.app = FastAPI(
            title="VigilAI Analytics Dashboard",
            description="Real-time analytics and insights for VigilAI",
            version="1.0.0"
        )
        
        # Setup templates
        self.templates = Jinja2Templates(directory="templates")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup dashboard routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page"""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "VigilAI Analytics Dashboard"
            })
        
        @self.app.get("/api/overview")
        async def get_overview():
            """Get system overview metrics"""
            try:
                # Get key metrics
                metrics = await self._get_system_metrics()
                return {
                    "timestamp": datetime.utcnow(),
                    "metrics": metrics
                }
            except Exception as e:
                logger.error(f"Error getting overview: {e}")
                raise HTTPException(status_code=500, detail="Failed to get overview")
        
        @self.app.get("/api/fleet/status")
        async def get_fleet_status():
            """Get fleet status and health"""
            try:
                fleet_data = await self._get_fleet_data()
                return fleet_data
            except Exception as e:
                logger.error(f"Error getting fleet status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get fleet status")
        
        @self.app.get("/api/analytics/detection")
        async def get_detection_analytics(
            start_date: Optional[str] = Query(None),
            end_date: Optional[str] = Query(None),
            device_id: Optional[str] = Query(None)
        ):
            """Get detection analytics"""
            try:
                analytics = await self._get_detection_analytics(
                    start_date, end_date, device_id
                )
                return analytics
            except Exception as e:
                logger.error(f"Error getting detection analytics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get detection analytics")
        
        @self.app.get("/api/analytics/performance")
        async def get_performance_analytics():
            """Get system performance analytics"""
            try:
                performance = await self._get_performance_analytics()
                return performance
            except Exception as e:
                logger.error(f"Error getting performance analytics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get performance analytics")
        
        @self.app.get("/api/analytics/users")
        async def get_user_analytics():
            """Get user analytics and insights"""
            try:
                user_analytics = await self._get_user_analytics()
                return user_analytics
            except Exception as e:
                logger.error(f"Error getting user analytics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get user analytics")
        
        @self.app.get("/api/charts/fatigue-trend")
        async def get_fatigue_trend_chart():
            """Get fatigue detection trend chart"""
            try:
                chart_data = await self._get_fatigue_trend_data()
                return chart_data
            except Exception as e:
                logger.error(f"Error getting fatigue trend: {e}")
                raise HTTPException(status_code=500, detail="Failed to get fatigue trend")
        
        @self.app.get("/api/charts/stress-distribution")
        async def get_stress_distribution_chart():
            """Get stress level distribution chart"""
            try:
                chart_data = await self._get_stress_distribution_data()
                return chart_data
            except Exception as e:
                logger.error(f"Error getting stress distribution: {e}")
                raise HTTPException(status_code=500, detail="Failed to get stress distribution")
        
        @self.app.get("/api/charts/device-health")
        async def get_device_health_chart():
            """Get device health chart"""
            try:
                chart_data = await self._get_device_health_data()
                return chart_data
            except Exception as e:
                logger.error(f"Error getting device health: {e}")
                raise HTTPException(status_code=500, detail="Failed to get device health")
        
        @self.app.get("/api/reports/daily")
        async def generate_daily_report(
            date: Optional[str] = Query(None)
        ):
            """Generate daily analytics report"""
            try:
                report_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
                report = await self._generate_daily_report(report_date)
                return report
            except Exception as e:
                logger.error(f"Error generating daily report: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate daily report")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "analytics_dashboard"}
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system overview metrics"""
        try:
            # Get metrics from database
            with engine.connect() as conn:
                # Total users
                user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                
                # Total devices
                device_count = conn.execute(text("SELECT COUNT(*) FROM devices")).scalar()
                
                # Active devices (last 24 hours)
                active_devices = conn.execute(text("""
                    SELECT COUNT(DISTINCT device_id) 
                    FROM device_health 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                """)).scalar()
                
                # Total detections today
                detections_today = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM processing_results 
                    WHERE DATE(timestamp) = CURRENT_DATE
                """)).scalar()
                
                # System uptime
                uptime = conn.execute(text("""
                    SELECT EXTRACT(EPOCH FROM (NOW() - MIN(created_at))) / 3600 
                    FROM users
                """)).scalar()
                
                return {
                    "total_users": user_count,
                    "total_devices": device_count,
                    "active_devices": active_devices,
                    "detections_today": detections_today,
                    "system_uptime_hours": uptime,
                    "timestamp": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def _get_fleet_data(self) -> Dict[str, Any]:
        """Get fleet status and health data"""
        try:
            with engine.connect() as conn:
                # Device status distribution
                status_dist = conn.execute(text("""
                    SELECT status, COUNT(*) as count
                    FROM devices
                    GROUP BY status
                """)).fetchall()
                
                # Device health metrics
                health_metrics = conn.execute(text("""
                    SELECT 
                        AVG(cpu_usage) as avg_cpu,
                        AVG(memory_usage) as avg_memory,
                        AVG(network_latency) as avg_latency,
                        COUNT(*) as total_health_records
                    FROM device_health
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)).fetchone()
                
                # Offline devices
                offline_devices = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM devices 
                    WHERE last_seen < NOW() - INTERVAL '5 minutes'
                """)).scalar()
                
                return {
                    "status_distribution": dict(status_dist),
                    "health_metrics": {
                        "avg_cpu": float(health_metrics.avg_cpu or 0),
                        "avg_memory": float(health_metrics.avg_memory or 0),
                        "avg_latency": float(health_metrics.avg_latency or 0),
                        "total_records": health_metrics.total_health_records
                    },
                    "offline_devices": offline_devices,
                    "timestamp": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting fleet data: {e}")
            return {}
    
    async def _get_detection_analytics(self, start_date: Optional[str], end_date: Optional[str], device_id: Optional[str]) -> Dict[str, Any]:
        """Get detection analytics"""
        try:
            with engine.connect() as conn:
                # Build query conditions
                conditions = []
                params = {}
                
                if start_date:
                    conditions.append("timestamp >= :start_date")
                    params["start_date"] = start_date
                
                if end_date:
                    conditions.append("timestamp <= :end_date")
                    params["end_date"] = end_date
                
                if device_id:
                    conditions.append("device_id = :device_id")
                    params["device_id"] = device_id
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Detection counts by type
                detection_counts = conn.execute(text(f"""
                    SELECT result_type, COUNT(*) as count
                    FROM processing_results
                    WHERE {where_clause}
                    GROUP BY result_type
                """), params).fetchall()
                
                # Confidence distribution
                confidence_stats = conn.execute(text(f"""
                    SELECT 
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        STDDEV(confidence) as std_confidence
                    FROM processing_results
                    WHERE {where_clause}
                """), params).fetchone()
                
                # Time series data
                time_series = conn.execute(text(f"""
                    SELECT 
                        DATE_TRUNC('hour', timestamp) as hour,
                        COUNT(*) as detections,
                        AVG(confidence) as avg_confidence
                    FROM processing_results
                    WHERE {where_clause}
                    GROUP BY hour
                    ORDER BY hour
                """), params).fetchall()
                
                return {
                    "detection_counts": dict(detection_counts),
                    "confidence_stats": {
                        "avg": float(confidence_stats.avg_confidence or 0),
                        "min": float(confidence_stats.min_confidence or 0),
                        "max": float(confidence_stats.max_confidence or 0),
                        "std": float(confidence_stats.std_confidence or 0)
                    },
                    "time_series": [
                        {
                            "hour": row.hour.isoformat(),
                            "detections": row.detections,
                            "avg_confidence": float(row.avg_confidence or 0)
                        }
                        for row in time_series
                    ],
                    "timestamp": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting detection analytics: {e}")
            return {}
    
    async def _get_performance_analytics(self) -> Dict[str, Any]:
        """Get system performance analytics"""
        try:
            # Get performance metrics from Redis
            metrics = {
                "api_response_time": redis_client.get("metrics:api_response_time") or "0",
                "processing_latency": redis_client.get("metrics:processing_latency") or "0",
                "throughput": redis_client.get("metrics:throughput") or "0",
                "error_rate": redis_client.get("metrics:error_rate") or "0"
            }
            
            # Convert to float
            for key, value in metrics.items():
                try:
                    metrics[key] = float(value)
                except (ValueError, TypeError):
                    metrics[key] = 0.0
            
            return {
                "performance_metrics": metrics,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {}
    
    async def _get_user_analytics(self) -> Dict[str, Any]:
        """Get user analytics and insights"""
        try:
            with engine.connect() as conn:
                # User registration trends
                registration_trends = conn.execute(text("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as registrations
                    FROM users
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY date
                    ORDER BY date
                """)).fetchall()
                
                # User activity
                active_users = conn.execute(text("""
                    SELECT COUNT(DISTINCT user_id)
                    FROM processing_results
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                """)).scalar()
                
                # User engagement
                engagement_metrics = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT user_id) as total_users,
                        COUNT(DISTINCT device_id) as total_devices,
                        AVG(daily_sessions) as avg_daily_sessions
                    FROM (
                        SELECT 
                            user_id,
                            device_id,
                            COUNT(DISTINCT DATE(timestamp)) as daily_sessions
                        FROM processing_results
                        WHERE timestamp > NOW() - INTERVAL '7 days'
                        GROUP BY user_id, device_id
                    ) user_activity
                """)).fetchone()
                
                return {
                    "registration_trends": [
                        {
                            "date": row.date.isoformat(),
                            "registrations": row.registrations
                        }
                        for row in registration_trends
                    ],
                    "active_users_24h": active_users,
                    "engagement_metrics": {
                        "total_users": engagement_metrics.total_users,
                        "total_devices": engagement_metrics.total_devices,
                        "avg_daily_sessions": float(engagement_metrics.avg_daily_sessions or 0)
                    },
                    "timestamp": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting user analytics: {e}")
            return {}
    
    async def _get_fatigue_trend_data(self) -> Dict[str, Any]:
        """Get fatigue detection trend data for chart"""
        try:
            with engine.connect() as conn:
                # Get fatigue detection trends over time
                trend_data = conn.execute(text("""
                    SELECT 
                        DATE_TRUNC('hour', timestamp) as hour,
                        COUNT(*) as fatigue_detections,
                        AVG(confidence) as avg_confidence
                    FROM processing_results
                    WHERE result_type = 'fatigue_detection'
                    AND timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY hour
                    ORDER BY hour
                """)).fetchall()
                
                # Create Plotly chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=[row.hour for row in trend_data],
                    y=[row.fatigue_detections for row in trend_data],
                    mode='lines+markers',
                    name='Fatigue Detections',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title="Fatigue Detection Trends (24h)",
                    xaxis_title="Time",
                    yaxis_title="Detections",
                    hovermode='x unified'
                )
                
                return {
                    "chart_html": fig.to_html(include_plotlyjs=False),
                    "data": [
                        {
                            "hour": row.hour.isoformat(),
                            "detections": row.fatigue_detections,
                            "avg_confidence": float(row.avg_confidence or 0)
                        }
                        for row in trend_data
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting fatigue trend data: {e}")
            return {}
    
    async def _get_stress_distribution_data(self) -> Dict[str, Any]:
        """Get stress level distribution data for chart"""
        try:
            with engine.connect() as conn:
                # Get stress level distribution
                stress_data = conn.execute(text("""
                    SELECT 
                        CASE 
                            WHEN confidence < 0.3 THEN 'Low'
                            WHEN confidence < 0.7 THEN 'Medium'
                            ELSE 'High'
                        END as stress_level,
                        COUNT(*) as count
                    FROM processing_results
                    WHERE result_type = 'stress_detection'
                    AND timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY stress_level
                """)).fetchall()
                
                # Create Plotly pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=[row.stress_level for row in stress_data],
                    values=[row.count for row in stress_data],
                    hole=0.3
                )])
                
                fig.update_layout(
                    title="Stress Level Distribution (24h)",
                    annotations=[dict(text='Stress Levels', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                
                return {
                    "chart_html": fig.to_html(include_plotlyjs=False),
                    "data": dict(stress_data)
                }
                
        except Exception as e:
            logger.error(f"Error getting stress distribution data: {e}")
            return {}
    
    async def _get_device_health_data(self) -> Dict[str, Any]:
        """Get device health data for chart"""
        try:
            with engine.connect() as conn:
                # Get device health metrics
                health_data = conn.execute(text("""
                    SELECT 
                        device_id,
                        AVG(cpu_usage) as avg_cpu,
                        AVG(memory_usage) as avg_memory,
                        AVG(network_latency) as avg_latency,
                        COUNT(*) as health_records
                    FROM device_health
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                    GROUP BY device_id
                    ORDER BY avg_cpu DESC
                    LIMIT 20
                """)).fetchall()
                
                # Create Plotly bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[row.device_id for row in health_data],
                    y=[row.avg_cpu for row in health_data],
                    name='CPU Usage %',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Device CPU Usage (Top 20 Devices)",
                    xaxis_title="Device ID",
                    yaxis_title="CPU Usage %",
                    xaxis={'tickangle': 45}
                )
                
                return {
                    "chart_html": fig.to_html(include_plotlyjs=False),
                    "data": [
                        {
                            "device_id": row.device_id,
                            "avg_cpu": float(row.avg_cpu or 0),
                            "avg_memory": float(row.avg_memory or 0),
                            "avg_latency": float(row.avg_latency or 0),
                            "health_records": row.health_records
                        }
                        for row in health_data
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting device health data: {e}")
            return {}
    
    async def _generate_daily_report(self, date: datetime) -> Dict[str, Any]:
        """Generate daily analytics report"""
        try:
            with engine.connect() as conn:
                # Daily summary
                daily_summary = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_detections,
                        COUNT(DISTINCT device_id) as active_devices,
                        COUNT(DISTINCT user_id) as active_users,
                        AVG(confidence) as avg_confidence
                    FROM processing_results
                    WHERE DATE(timestamp) = :date
                """), {"date": date.date()}).fetchone()
                
                # Detection breakdown
                detection_breakdown = conn.execute(text("""
                    SELECT 
                        result_type,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM processing_results
                    WHERE DATE(timestamp) = :date
                    GROUP BY result_type
                """), {"date": date.date()}).fetchall()
                
                # Top performing devices
                top_devices = conn.execute(text("""
                    SELECT 
                        device_id,
                        COUNT(*) as detections,
                        AVG(confidence) as avg_confidence
                    FROM processing_results
                    WHERE DATE(timestamp) = :date
                    GROUP BY device_id
                    ORDER BY detections DESC
                    LIMIT 10
                """), {"date": date.date()}).fetchall()
                
                return {
                    "date": date.isoformat(),
                    "summary": {
                        "total_detections": daily_summary.total_detections,
                        "active_devices": daily_summary.active_devices,
                        "active_users": daily_summary.active_users,
                        "avg_confidence": float(daily_summary.avg_confidence or 0)
                    },
                    "detection_breakdown": [
                        {
                            "type": row.result_type,
                            "count": row.count,
                            "avg_confidence": float(row.avg_confidence or 0)
                        }
                        for row in detection_breakdown
                    ],
                    "top_devices": [
                        {
                            "device_id": row.device_id,
                            "detections": row.detections,
                            "avg_confidence": float(row.avg_confidence or 0)
                        }
                        for row in top_devices
                    ],
                    "generated_at": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {}

def create_analytics_dashboard() -> AnalyticsDashboard:
    """Create analytics dashboard instance"""
    return AnalyticsDashboard()

def main():
    """Main entry point for analytics dashboard"""
    dashboard = create_analytics_dashboard()
    
    uvicorn.run(
        dashboard.app,
        host="0.0.0.0",
        port=8007,
        log_level="info"
    )

if __name__ == "__main__":
    main()
