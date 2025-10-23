-- VigilAI Database Initialization Script
-- Creates all necessary tables and indexes for the VigilAI system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    role VARCHAR(50) DEFAULT 'user',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en'
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(500) UNIQUE NOT NULL,
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Devices table
CREATE TABLE IF NOT EXISTS devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    device_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    fleet_id UUID,
    location JSONB,
    capabilities JSONB,
    firmware_version VARCHAR(50),
    hardware_info JSONB,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Device health table
CREATE TABLE IF NOT EXISTS device_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    disk_usage DECIMAL(5,2),
    network_latency DECIMAL(8,3),
    temperature DECIMAL(5,2),
    battery_level DECIMAL(5,2),
    signal_strength DECIMAL(5,2),
    is_online BOOLEAN DEFAULT TRUE,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    performance_metrics JSONB
);

-- Processing results table
CREATE TABLE IF NOT EXISTS processing_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    result_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB,
    processing_time DECIMAL(8,3),
    model_version VARCHAR(50),
    synced BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fleets table
CREATE TABLE IF NOT EXISTS fleets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Interventions table
CREATE TABLE IF NOT EXISTS interventions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    intervention_type VARCHAR(50) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    trigger_value DECIMAL(5,4),
    intervention_data JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    effectiveness_score DECIMAL(5,4)
);

-- Analytics events table
CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    device_id UUID REFERENCES devices(id) ON DELETE SET NULL,
    session_id VARCHAR(100),
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_devices_device_id ON devices(device_id);
CREATE INDEX IF NOT EXISTS idx_devices_owner_id ON devices(owner_id);
CREATE INDEX IF NOT EXISTS idx_devices_fleet_id ON devices(fleet_id);
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_device_health_device_id ON device_health(device_id);
CREATE INDEX IF NOT EXISTS idx_device_health_timestamp ON device_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_processing_results_device_id ON processing_results(device_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_user_id ON processing_results(user_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_timestamp ON processing_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_processing_results_type ON processing_results(result_type);
CREATE INDEX IF NOT EXISTS idx_processing_results_synced ON processing_results(synced);
CREATE INDEX IF NOT EXISTS idx_alerts_device_id ON alerts(device_id);
CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(is_resolved);
CREATE INDEX IF NOT EXISTS idx_interventions_device_id ON interventions(device_id);
CREATE INDEX IF NOT EXISTS idx_interventions_user_id ON interventions(user_id);
CREATE INDEX IF NOT EXISTS idx_interventions_type ON interventions(intervention_type);
CREATE INDEX IF NOT EXISTS idx_interventions_active ON interventions(is_active);
CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_device_id ON analytics_events(device_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_timestamp ON analytics_events(timestamp);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_users_preferences_gin ON users USING gin(preferences);
CREATE INDEX IF NOT EXISTS idx_devices_location_gin ON devices USING gin(location);
CREATE INDEX IF NOT EXISTS idx_devices_capabilities_gin ON devices USING gin(capabilities);
CREATE INDEX IF NOT EXISTS idx_devices_metadata_gin ON devices USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_device_health_performance_gin ON device_health USING gin(performance_metrics);
CREATE INDEX IF NOT EXISTS idx_processing_results_data_gin ON processing_results USING gin(data);
CREATE INDEX IF NOT EXISTS idx_processing_results_metadata_gin ON processing_results USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_alerts_data_gin ON alerts USING gin(data);
CREATE INDEX IF NOT EXISTS idx_interventions_data_gin ON interventions USING gin(intervention_data);
CREATE INDEX IF NOT EXISTS idx_analytics_events_data_gin ON analytics_events USING gin(event_data);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_processing_results_device_timestamp ON processing_results(device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_processing_results_user_timestamp ON processing_results(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_device_health_device_timestamp ON device_health(device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_device_created ON alerts(device_id, created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_user_created ON alerts(user_id, created_at);

-- Create partial indexes for active records
CREATE INDEX IF NOT EXISTS idx_users_active_verified ON users(is_active, is_verified) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_user_sessions_active_token ON user_sessions(token) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_devices_active ON devices(device_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_processing_results_unsynced ON processing_results(id) WHERE synced = FALSE;
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON alerts(device_id, created_at) WHERE is_resolved = FALSE;
CREATE INDEX IF NOT EXISTS idx_interventions_active ON interventions(device_id, started_at) WHERE is_active = TRUE;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_devices_updated_at BEFORE UPDATE ON devices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fleets_updated_at BEFORE UPDATE ON fleets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean up old sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get user statistics
CREATE OR REPLACE FUNCTION get_user_stats(user_uuid UUID)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_devices', COUNT(d.id),
        'active_devices', COUNT(d.id) FILTER (WHERE d.status = 'active'),
        'total_alerts', COUNT(a.id),
        'unresolved_alerts', COUNT(a.id) FILTER (WHERE a.is_resolved = FALSE),
        'total_interventions', COUNT(i.id),
        'active_interventions', COUNT(i.id) FILTER (WHERE i.is_active = TRUE),
        'last_activity', MAX(d.last_seen)
    ) INTO result
    FROM devices d
    LEFT JOIN alerts a ON a.device_id = d.id
    LEFT JOIN interventions i ON i.device_id = d.id
    WHERE d.owner_id = user_uuid;
    
    RETURN COALESCE(result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Insert default admin user (password: admin123)
INSERT INTO users (email, username, hashed_password, full_name, role, is_active, is_verified)
VALUES (
    'admin@vigilai.com',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/9KzKz2K', -- bcrypt hash of 'admin123'
    'System Administrator',
    'admin',
    TRUE,
    TRUE
) ON CONFLICT (email) DO NOTHING;

-- Create view for device statistics
CREATE OR REPLACE VIEW device_stats AS
SELECT 
    d.id,
    d.device_id,
    d.name,
    d.device_type,
    d.status,
    d.owner_id,
    u.username as owner_username,
    d.last_seen,
    COALESCE(h.cpu_usage, 0) as last_cpu_usage,
    COALESCE(h.memory_usage, 0) as last_memory_usage,
    COALESCE(h.is_online, FALSE) as is_online,
    COUNT(a.id) as total_alerts,
    COUNT(a.id) FILTER (WHERE a.is_resolved = FALSE) as unresolved_alerts,
    COUNT(i.id) as total_interventions,
    COUNT(i.id) FILTER (WHERE i.is_active = TRUE) as active_interventions
FROM devices d
LEFT JOIN users u ON u.id = d.owner_id
LEFT JOIN device_health h ON h.device_id = d.id AND h.timestamp = (
    SELECT MAX(timestamp) FROM device_health WHERE device_id = d.id
)
LEFT JOIN alerts a ON a.device_id = d.id
LEFT JOIN interventions i ON i.device_id = d.id
GROUP BY d.id, d.device_id, d.name, d.device_type, d.status, d.owner_id, u.username, d.last_seen, h.cpu_usage, h.memory_usage, h.is_online;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vigilai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vigilai;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO vigilai;
