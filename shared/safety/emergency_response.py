"""
Emergency Response System for VigilAI
Advanced safety features including crash detection, emergency alerts, and automated response
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import twilio
from twilio.rest import Client
import numpy as np
from scipy import signal
import cv2

logger = logging.getLogger(__name__)

class EmergencyType(Enum):
    """Types of emergency situations"""
    CRASH = "crash"
    MEDICAL = "medical"
    FATIGUE_CRITICAL = "fatigue_critical"
    STRESS_CRITICAL = "stress_critical"
    SYSTEM_FAILURE = "system_failure"
    MANUAL = "manual"

class EmergencySeverity(Enum):
    """Emergency severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EmergencyEvent:
    """Emergency event data"""
    event_id: str
    emergency_type: EmergencyType
    severity: EmergencySeverity
    timestamp: datetime
    location: Dict[str, float]  # lat, lng
    device_id: str
    user_id: str
    description: str
    sensor_data: Dict[str, Any]
    auto_triggered: bool
    response_actions: List[str]
    status: str = "active"

@dataclass
class EmergencyContact:
    """Emergency contact information"""
    name: str
    phone: str
    email: str
    relationship: str
    priority: int  # 1 = highest priority
    is_primary: bool = False

class CrashDetector:
    """Advanced crash detection using multiple sensors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.acceleration_threshold = config.get('acceleration_threshold', 2.5)  # g-force
        self.impact_duration = config.get('impact_duration', 0.1)  # seconds
        self.velocity_change_threshold = config.get('velocity_change_threshold', 15)  # m/s
        
        # Sensor data buffers
        self.acceleration_buffer = []
        self.velocity_buffer = []
        self.orientation_buffer = []
        self.audio_buffer = []
        
        # Crash detection state
        self.crash_detected = False
        self.crash_confidence = 0.0
        self.last_crash_time = None
        
    async def analyze_sensor_data(self, sensor_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Analyze sensor data for crash indicators"""
        try:
            # Extract sensor data
            acceleration = sensor_data.get('acceleration', {})
            velocity = sensor_data.get('velocity', {})
            orientation = sensor_data.get('orientation', {})
            audio = sensor_data.get('audio', {})
            
            # Update buffers
            self._update_buffers(acceleration, velocity, orientation, audio)
            
            # Check for crash indicators
            crash_indicators = []
            
            # 1. Sudden acceleration/deceleration
            if len(self.acceleration_buffer) >= 10:
                accel_magnitude = np.linalg.norm(self.acceleration_buffer[-1])
                if accel_magnitude > self.acceleration_threshold:
                    crash_indicators.append(('acceleration', 0.8))
            
            # 2. Rapid velocity change
            if len(self.velocity_buffer) >= 5:
                velocity_change = abs(self.velocity_buffer[-1] - self.velocity_buffer[-5])
                if velocity_change > self.velocity_change_threshold:
                    crash_indicators.append(('velocity_change', 0.7))
            
            # 3. Orientation change (rollover detection)
            if len(self.orientation_buffer) >= 5:
                orientation_change = self._calculate_orientation_change()
                if orientation_change > 45:  # degrees
                    crash_indicators.append(('orientation', 0.9))
            
            # 4. Audio analysis (crash sound detection)
            if audio.get('volume', 0) > 0.8:  # High volume threshold
                crash_indicators.append(('audio', 0.6))
            
            # 5. Pattern analysis
            pattern_score = self._analyze_crash_pattern()
            if pattern_score > 0.7:
                crash_indicators.append(('pattern', pattern_score))
            
            # Calculate overall crash confidence
            if crash_indicators:
                self.crash_confidence = max([score for _, score in crash_indicators])
                crash_detected = self.crash_confidence > 0.6
            else:
                self.crash_confidence = 0.0
                crash_detected = False
            
            # Update crash state
            if crash_detected and not self.crash_detected:
                self.crash_detected = True
                self.last_crash_time = datetime.utcnow()
                logger.warning(f"Crash detected with confidence: {self.crash_confidence}")
            
            return crash_detected, self.crash_confidence
            
        except Exception as e:
            logger.error(f"Error analyzing sensor data: {e}")
            return False, 0.0
    
    def _update_buffers(self, acceleration: Dict, velocity: Dict, orientation: Dict, audio: Dict):
        """Update sensor data buffers"""
        # Acceleration (3-axis)
        if acceleration:
            accel_vector = [acceleration.get('x', 0), acceleration.get('y', 0), acceleration.get('z', 0)]
            self.acceleration_buffer.append(accel_vector)
            if len(self.acceleration_buffer) > 50:  # Keep last 50 readings
                self.acceleration_buffer.pop(0)
        
        # Velocity
        if velocity:
            vel_magnitude = velocity.get('magnitude', 0)
            self.velocity_buffer.append(vel_magnitude)
            if len(self.velocity_buffer) > 20:
                self.velocity_buffer.pop(0)
        
        # Orientation (pitch, roll, yaw)
        if orientation:
            orient_vector = [orientation.get('pitch', 0), orientation.get('roll', 0), orientation.get('yaw', 0)]
            self.orientation_buffer.append(orient_vector)
            if len(self.orientation_buffer) > 20:
                self.orientation_buffer.pop(0)
        
        # Audio
        if audio:
            self.audio_buffer.append(audio.get('volume', 0))
            if len(self.audio_buffer) > 100:
                self.audio_buffer.pop(0)
    
    def _calculate_orientation_change(self) -> float:
        """Calculate orientation change for rollover detection"""
        if len(self.orientation_buffer) < 5:
            return 0.0
        
        current = np.array(self.orientation_buffer[-1])
        previous = np.array(self.orientation_buffer[-5])
        
        # Calculate angular difference
        diff = np.linalg.norm(current - previous)
        return float(diff)
    
    def _analyze_crash_pattern(self) -> float:
        """Analyze patterns in sensor data for crash indicators"""
        if len(self.acceleration_buffer) < 10:
            return 0.0
        
        # Convert to numpy array
        accel_data = np.array(self.acceleration_buffer)
        
        # Calculate magnitude of acceleration
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        
        # Look for sudden spikes
        if len(accel_magnitude) >= 5:
            recent_avg = np.mean(accel_magnitude[-5:])
            overall_avg = np.mean(accel_magnitude)
            
            if recent_avg > overall_avg * 2:  # Sudden increase
                return 0.8
        
        return 0.0

class EmergencyResponseSystem:
    """Comprehensive emergency response system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crash_detector = CrashDetector(config.get('crash_detection', {}))
        self.emergency_contacts = []
        self.active_emergencies = []
        
        # Communication services
        self.twilio_client = None
        self.email_config = config.get('email', {})
        self.api_endpoints = config.get('api_endpoints', {})
        
        # Initialize communication services
        self._initialize_communication_services()
        
        # Load emergency contacts
        self._load_emergency_contacts()
    
    def _initialize_communication_services(self):
        """Initialize communication services"""
        try:
            # Initialize Twilio for SMS/calls
            twilio_config = self.config.get('twilio', {})
            if twilio_config.get('account_sid') and twilio_config.get('auth_token'):
                self.twilio_client = Client(
                    twilio_config['account_sid'],
                    twilio_config['auth_token']
                )
            
            logger.info("Communication services initialized")
        except Exception as e:
            logger.error(f"Error initializing communication services: {e}")
    
    def _load_emergency_contacts(self):
        """Load emergency contacts from configuration"""
        contacts_config = self.config.get('emergency_contacts', [])
        
        for contact_data in contacts_config:
            contact = EmergencyContact(
                name=contact_data['name'],
                phone=contact_data['phone'],
                email=contact_data.get('email', ''),
                relationship=contact_data.get('relationship', ''),
                priority=contact_data.get('priority', 1),
                is_primary=contact_data.get('is_primary', False)
            )
            self.emergency_contacts.append(contact)
    
    async def monitor_for_emergencies(self, sensor_data: Dict[str, Any], user_data: Dict[str, Any]) -> List[EmergencyEvent]:
        """Monitor sensor data for emergency situations"""
        emergencies = []
        
        try:
            # 1. Crash detection
            crash_detected, crash_confidence = await self.crash_detector.analyze_sensor_data(sensor_data)
            if crash_detected:
                emergency = await self._create_emergency_event(
                    EmergencyType.CRASH,
                    EmergencySeverity.CRITICAL,
                    "Vehicle crash detected",
                    sensor_data,
                    user_data,
                    auto_triggered=True
                )
                emergencies.append(emergency)
            
            # 2. Critical fatigue detection
            fatigue_level = sensor_data.get('fatigue_level', 0)
            if fatigue_level > 0.9:  # Critical fatigue
                emergency = await self._create_emergency_event(
                    EmergencyType.FATIGUE_CRITICAL,
                    EmergencySeverity.HIGH,
                    "Critical fatigue level detected",
                    sensor_data,
                    user_data,
                    auto_triggered=True
                )
                emergencies.append(emergency)
            
            # 3. Critical stress detection
            stress_level = sensor_data.get('stress_level', 0)
            if stress_level > 0.9:  # Critical stress
                emergency = await self._create_emergency_event(
                    EmergencyType.STRESS_CRITICAL,
                    EmergencySeverity.HIGH,
                    "Critical stress level detected",
                    sensor_data,
                    user_data,
                    auto_triggered=True
                )
                emergencies.append(emergency)
            
            # 4. Medical emergency detection (simplified)
            biometric_data = sensor_data.get('biometric', {})
            if self._detect_medical_emergency(biometric_data):
                emergency = await self._create_emergency_event(
                    EmergencyType.MEDICAL,
                    EmergencySeverity.CRITICAL,
                    "Medical emergency detected",
                    sensor_data,
                    user_data,
                    auto_triggered=True
                )
                emergencies.append(emergency)
            
            # Process any new emergencies
            for emergency in emergencies:
                await self._process_emergency(emergency)
                self.active_emergencies.append(emergency)
            
        except Exception as e:
            logger.error(f"Error monitoring for emergencies: {e}")
        
        return emergencies
    
    def _detect_medical_emergency(self, biometric_data: Dict[str, Any]) -> bool:
        """Detect potential medical emergencies from biometric data"""
        try:
            # Check for abnormal heart rate
            heart_rate = biometric_data.get('heart_rate', 70)
            if heart_rate < 40 or heart_rate > 150:  # Abnormal heart rate
                return True
            
            # Check for loss of consciousness indicators
            hrv = biometric_data.get('hrv', 0.05)
            if hrv < 0.02:  # Very low HRV might indicate unconsciousness
                return True
            
            # Check for temperature abnormalities
            temperature = biometric_data.get('temperature', 36.5)
            if temperature < 35.0 or temperature > 39.0:  # Hypothermia or fever
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting medical emergency: {e}")
            return False
    
    async def _create_emergency_event(self, emergency_type: EmergencyType, severity: EmergencySeverity,
                                    description: str, sensor_data: Dict, user_data: Dict, 
                                    auto_triggered: bool = True) -> EmergencyEvent:
        """Create an emergency event"""
        event_id = f"emergency_{int(time.time())}_{emergency_type.value}"
        
        # Get location from sensor data or user data
        location = sensor_data.get('location', user_data.get('location', {'lat': 0, 'lng': 0}))
        
        # Determine response actions based on emergency type and severity
        response_actions = self._determine_response_actions(emergency_type, severity)
        
        return EmergencyEvent(
            event_id=event_id,
            emergency_type=emergency_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            location=location,
            device_id=user_data.get('device_id', 'unknown'),
            user_id=user_data.get('user_id', 'unknown'),
            description=description,
            sensor_data=sensor_data,
            auto_triggered=auto_triggered,
            response_actions=response_actions
        )
    
    def _determine_response_actions(self, emergency_type: EmergencyType, severity: EmergencySeverity) -> List[str]:
        """Determine appropriate response actions"""
        actions = []
        
        if emergency_type == EmergencyType.CRASH:
            actions.extend([
                'call_emergency_services',
                'notify_emergency_contacts',
                'send_location_to_authorities',
                'activate_vehicle_safety_systems'
            ])
        elif emergency_type == EmergencyType.MEDICAL:
            actions.extend([
                'call_emergency_services',
                'notify_emergency_contacts',
                'send_medical_info_to_authorities',
                'guide_to_nearest_hospital'
            ])
        elif emergency_type in [EmergencyType.FATIGUE_CRITICAL, EmergencyType.STRESS_CRITICAL]:
            actions.extend([
                'notify_emergency_contacts',
                'suggest_immediate_break',
                'guide_to_safe_location',
                'monitor_continuously'
            ])
        
        # Add severity-based actions
        if severity == EmergencySeverity.CRITICAL:
            actions.append('immediate_emergency_response')
        elif severity == EmergencySeverity.HIGH:
            actions.append('urgent_response')
        
        return actions
    
    async def _process_emergency(self, emergency: EmergencyEvent):
        """Process an emergency event"""
        try:
            logger.warning(f"Processing emergency: {emergency.emergency_type.value} - {emergency.description}")
            
            # Execute response actions
            for action in emergency.response_actions:
                await self._execute_response_action(action, emergency)
            
            # Log emergency
            await self._log_emergency(emergency)
            
        except Exception as e:
            logger.error(f"Error processing emergency: {e}")
    
    async def _execute_response_action(self, action: str, emergency: EmergencyEvent):
        """Execute a specific response action"""
        try:
            if action == 'call_emergency_services':
                await self._call_emergency_services(emergency)
            elif action == 'notify_emergency_contacts':
                await self._notify_emergency_contacts(emergency)
            elif action == 'send_location_to_authorities':
                await self._send_location_to_authorities(emergency)
            elif action == 'send_medical_info_to_authorities':
                await self._send_medical_info_to_authorities(emergency)
            elif action == 'guide_to_nearest_hospital':
                await self._guide_to_nearest_hospital(emergency)
            elif action == 'suggest_immediate_break':
                await self._suggest_immediate_break(emergency)
            elif action == 'guide_to_safe_location':
                await self._guide_to_safe_location(emergency)
            elif action == 'monitor_continuously':
                await self._monitor_continuously(emergency)
            elif action == 'immediate_emergency_response':
                await self._immediate_emergency_response(emergency)
            elif action == 'urgent_response':
                await self._urgent_response(emergency)
            
        except Exception as e:
            logger.error(f"Error executing response action {action}: {e}")
    
    async def _call_emergency_services(self, emergency: EmergencyEvent):
        """Call emergency services"""
        try:
            if self.twilio_client:
                # Make emergency call
                call = self.twilio_client.calls.create(
                    to=self.config.get('emergency_number', '911'),
                    from_=self.config.get('twilio', {}).get('phone_number'),
                    url='http://demo.twilio.com/docs/voice.xml'  # Emergency message URL
                )
                logger.info(f"Emergency call initiated: {call.sid}")
            
            # Send emergency data to authorities
            await self._send_emergency_data_to_authorities(emergency)
            
        except Exception as e:
            logger.error(f"Error calling emergency services: {e}")
    
    async def _notify_emergency_contacts(self, emergency: EmergencyEvent):
        """Notify emergency contacts"""
        try:
            # Sort contacts by priority
            sorted_contacts = sorted(self.emergency_contacts, key=lambda x: x.priority)
            
            for contact in sorted_contacts:
                # Send SMS
                if contact.phone and self.twilio_client:
                    await self._send_emergency_sms(contact, emergency)
                
                # Send email
                if contact.email:
                    await self._send_emergency_email(contact, emergency)
                
                # Add delay between notifications
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error notifying emergency contacts: {e}")
    
    async def _send_emergency_sms(self, contact: EmergencyContact, emergency: EmergencyEvent):
        """Send emergency SMS to contact"""
        try:
            message_body = f"""
🚨 EMERGENCY ALERT 🚨

{emergency.description}

Location: {emergency.location.get('lat', 0)}, {emergency.location.get('lng', 0)}
Time: {emergency.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Device: {emergency.device_id}

Please check on the driver immediately.
            """.strip()
            
            if self.twilio_client:
                message = self.twilio_client.messages.create(
                    body=message_body,
                    from_=self.config.get('twilio', {}).get('phone_number'),
                    to=contact.phone
                )
                logger.info(f"Emergency SMS sent to {contact.name}: {message.sid}")
            
        except Exception as e:
            logger.error(f"Error sending emergency SMS: {e}")
    
    async def _send_emergency_email(self, contact: EmergencyContact, emergency: EmergencyEvent):
        """Send emergency email to contact"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config.get('from_email', 'noreply@vigilai.com')
            msg['To'] = contact.email
            msg['Subject'] = f"🚨 EMERGENCY ALERT - {emergency.emergency_type.value.upper()}"
            
            body = f"""
Dear {contact.name},

This is an automated emergency alert from VigilAI.

EMERGENCY DETAILS:
- Type: {emergency.emergency_type.value}
- Severity: {emergency.severity.value}
- Description: {emergency.description}
- Location: {emergency.location.get('lat', 0)}, {emergency.location.get('lng', 0)}
- Time: {emergency.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Device ID: {emergency.device_id}
- User ID: {emergency.user_id}

Please check on the driver immediately and contact emergency services if necessary.

This is an automated message from VigilAI Safety System.
            """.strip()
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email (simplified - would need proper SMTP setup)
            logger.info(f"Emergency email prepared for {contact.name}")
            
        except Exception as e:
            logger.error(f"Error sending emergency email: {e}")
    
    async def _send_emergency_data_to_authorities(self, emergency: EmergencyEvent):
        """Send emergency data to authorities API"""
        try:
            emergency_data = {
                'event_id': emergency.event_id,
                'emergency_type': emergency.emergency_type.value,
                'severity': emergency.severity.value,
                'location': emergency.location,
                'timestamp': emergency.timestamp.isoformat(),
                'device_id': emergency.device_id,
                'user_id': emergency.user_id,
                'description': emergency.description,
                'sensor_data': emergency.sensor_data
            }
            
            # Send to emergency services API
            api_endpoint = self.api_endpoints.get('emergency_services')
            if api_endpoint:
                response = requests.post(api_endpoint, json=emergency_data, timeout=10)
                logger.info(f"Emergency data sent to authorities: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error sending emergency data to authorities: {e}")
    
    async def _log_emergency(self, emergency: EmergencyEvent):
        """Log emergency event"""
        try:
            log_data = {
                'event_id': emergency.event_id,
                'emergency_type': emergency.emergency_type.value,
                'severity': emergency.severity.value,
                'timestamp': emergency.timestamp.isoformat(),
                'location': emergency.location,
                'device_id': emergency.device_id,
                'user_id': emergency.user_id,
                'description': emergency.description,
                'auto_triggered': emergency.auto_triggered,
                'response_actions': emergency.response_actions
            }
            
            # Log to file or database
            logger.warning(f"EMERGENCY LOGGED: {json.dumps(log_data)}")
            
        except Exception as e:
            logger.error(f"Error logging emergency: {e}")
    
    async def manual_emergency_trigger(self, user_id: str, device_id: str, emergency_type: str, description: str):
        """Manually trigger an emergency"""
        try:
            emergency = await self._create_emergency_event(
                EmergencyType.MANUAL,
                EmergencySeverity.HIGH,
                f"Manual emergency: {description}",
                {},
                {'user_id': user_id, 'device_id': device_id},
                auto_triggered=False
            )
            
            await self._process_emergency(emergency)
            self.active_emergencies.append(emergency)
            
            logger.info(f"Manual emergency triggered by user {user_id}")
            
        except Exception as e:
            logger.error(f"Error triggering manual emergency: {e}")
    
    def get_active_emergencies(self) -> List[EmergencyEvent]:
        """Get list of active emergencies"""
        return [e for e in self.active_emergencies if e.status == "active"]
    
    async def resolve_emergency(self, event_id: str, resolution_notes: str = ""):
        """Resolve an emergency event"""
        try:
            for emergency in self.active_emergencies:
                if emergency.event_id == event_id:
                    emergency.status = "resolved"
                    logger.info(f"Emergency {event_id} resolved: {resolution_notes}")
                    break
            
        except Exception as e:
            logger.error(f"Error resolving emergency: {e}")
    
    async def cleanup(self):
        """Cleanup emergency response system"""
        try:
            self.active_emergencies.clear()
            logger.info("Emergency response system cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
