"""
Accessibility Features for VigilAI
Comprehensive accessibility support for users with disabilities
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
import pygame
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import threading
import time

logger = logging.getLogger(__name__)

class AccessibilityMode(Enum):
    """Accessibility modes"""
    VISUAL_IMPAIRMENT = "visual_impairment"
    HEARING_IMPAIRMENT = "hearing_impairment"
    MOTOR_IMPAIRMENT = "motor_impairment"
    COGNITIVE_IMPAIRMENT = "cognitive_impairment"
    MULTIPLE_IMPAIRMENTS = "multiple_impairments"

class NotificationType(Enum):
    """Types of accessibility notifications"""
    AUDIO = "audio"
    VISUAL = "visual"
    HAPTIC = "haptic"
    TEXT = "text"
    BRAILLE = "braille"

@dataclass
class AccessibilitySettings:
    """User accessibility settings"""
    user_id: str
    accessibility_mode: AccessibilityMode
    text_size: str  # small, medium, large, extra_large
    contrast_level: str  # normal, high, extra_high
    color_scheme: str  # default, dark, light, high_contrast
    audio_feedback: bool
    voice_guidance: bool
    haptic_feedback: bool
    screen_reader: bool
    voice_commands: bool
    gesture_control: bool
    simplified_interface: bool
    language: str
    voice_speed: float
    voice_pitch: float
    haptic_intensity: str  # light, medium, strong

class TextToSpeechEngine:
    """Advanced Text-to-Speech engine with accessibility features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = None
        self.is_initialized = False
        self.voice_queue = []
        self.current_voice = None
        
    async def initialize(self) -> bool:
        """Initialize TTS engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.engine.getProperty('voices')
            if voices:
                # Select appropriate voice based on language
                for voice in voices:
                    if 'english' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Set default properties
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.8)  # Volume
            
            self.is_initialized = True
            logger.info("TTS engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            return False
    
    async def speak(self, text: str, settings: AccessibilitySettings):
        """Speak text with accessibility settings"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Configure voice based on user settings
            self.engine.setProperty('rate', int(150 * settings.voice_speed))
            self.engine.setProperty('volume', 0.8)
            
            # Add to voice queue for proper sequencing
            self.voice_queue.append((text, settings))
            
            # Process voice queue
            await self._process_voice_queue()
            
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
    
    async def _process_voice_queue(self):
        """Process voice queue sequentially"""
        while self.voice_queue:
            text, settings = self.voice_queue.pop(0)
            
            # Configure voice settings
            self.engine.setProperty('rate', int(150 * settings.voice_speed))
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
            # Small delay between messages
            await asyncio.sleep(0.5)
    
    async def speak_emergency_alert(self, alert_text: str, settings: AccessibilitySettings):
        """Speak emergency alert with enhanced audio"""
        try:
            # Use more urgent voice settings for emergencies
            original_rate = self.engine.getProperty('rate')
            original_volume = self.engine.getProperty('volume')
            
            # Increase speed and volume for emergencies
            self.engine.setProperty('rate', int(original_rate * 1.2))
            self.engine.setProperty('volume', 1.0)
            
            # Speak emergency message
            await self.speak(f"EMERGENCY ALERT: {alert_text}", settings)
            
            # Restore original settings
            self.engine.setProperty('rate', original_rate)
            self.engine.setProperty('volume', original_volume)
            
        except Exception as e:
            logger.error(f"Error speaking emergency alert: {e}")
    
    async def speak_driving_instructions(self, instructions: str, settings: AccessibilitySettings):
        """Speak driving instructions with clear pronunciation"""
        try:
            # Use slower, clearer speech for instructions
            original_rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', int(original_rate * 0.8))
            
            await self.speak(f"Driving instruction: {instructions}", settings)
            
            # Restore original rate
            self.engine.setProperty('rate', original_rate)
            
        except Exception as e:
            logger.error(f"Error speaking driving instructions: {e}")

class SpeechRecognitionEngine:
    """Advanced Speech Recognition for voice commands"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_initialized = False
        self.voice_commands = {
            'emergency': ['emergency', 'help', 'sos', 'accident'],
            'fatigue': ['tired', 'sleepy', 'fatigue', 'drowsy'],
            'stress': ['stressed', 'anxious', 'nervous', 'panic'],
            'navigation': ['navigate', 'directions', 'route', 'map'],
            'settings': ['settings', 'configure', 'setup', 'options'],
            'status': ['status', 'report', 'check', 'monitor']
        }
    
    async def initialize(self) -> bool:
        """Initialize speech recognition"""
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.is_initialized = True
            logger.info("Speech recognition initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
            return False
    
    async def listen_for_commands(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for voice commands"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout)
            
            # Recognize speech
            try:
                text = self.recognizer.recognize_google(audio).lower()
                logger.info(f"Recognized speech: {text}")
                return text
            except sr.UnknownValueError:
                logger.debug("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error listening for commands: {e}")
            return None
    
    def parse_voice_command(self, text: str) -> Tuple[str, str]:
        """Parse voice command and return action and parameters"""
        try:
            text_lower = text.lower()
            
            # Check for emergency commands
            for command in self.voice_commands['emergency']:
                if command in text_lower:
                    return 'emergency', text_lower
            
            # Check for fatigue commands
            for command in self.voice_commands['fatigue']:
                if command in text_lower:
                    return 'fatigue', text_lower
            
            # Check for stress commands
            for command in self.voice_commands['stress']:
                if command in text_lower:
                    return 'stress', text_lower
            
            # Check for navigation commands
            for command in self.voice_commands['navigation']:
                if command in text_lower:
                    return 'navigation', text_lower
            
            # Check for settings commands
            for command in self.voice_commands['settings']:
                if command in text_lower:
                    return 'settings', text_lower
            
            # Check for status commands
            for command in self.voice_commands['status']:
                if command in text_lower:
                    return 'status', text_lower
            
            return 'unknown', text_lower
            
        except Exception as e:
            logger.error(f"Error parsing voice command: {e}")
            return 'error', text

class VisualAccessibilityEngine:
    """Visual accessibility features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize visual accessibility engine"""
        try:
            # Initialize pygame for visual feedback
            pygame.init()
            
            self.is_initialized = True
            logger.info("Visual accessibility engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing visual accessibility: {e}")
            return False
    
    def enhance_image_for_visual_impairment(self, image: np.ndarray, settings: AccessibilitySettings) -> np.ndarray:
        """Enhance image for users with visual impairments"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply contrast enhancement
            if settings.contrast_level == 'high':
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(2.0)
            elif settings.contrast_level == 'extra_high':
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(3.0)
            
            # Apply brightness enhancement
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            # Apply sharpening for better visibility
            pil_image = pil_image.filter(ImageFilter.SHARPEN)
            
            # Convert back to OpenCV format
            enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def create_high_contrast_overlay(self, image: np.ndarray, settings: AccessibilitySettings) -> np.ndarray:
        """Create high contrast overlay for better visibility"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply high contrast transformation
            if settings.color_scheme == 'high_contrast':
                # Create high contrast version
                high_contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
                
                # Convert back to BGR
                high_contrast_bgr = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
                
                return high_contrast_bgr
            
            return image
            
        except Exception as e:
            logger.error(f"Error creating high contrast overlay: {e}")
            return image
    
    def add_visual_indicators(self, image: np.ndarray, indicators: Dict[str, Any]) -> np.ndarray:
        """Add visual indicators for accessibility"""
        try:
            # Add fatigue level indicator
            if 'fatigue_level' in indicators:
                fatigue_level = indicators['fatigue_level']
                color = (0, 255, 0) if fatigue_level < 0.3 else (0, 255, 255) if fatigue_level < 0.7 else (0, 0, 255)
                cv2.putText(image, f"Fatigue: {fatigue_level:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add stress level indicator
            if 'stress_level' in indicators:
                stress_level = indicators['stress_level']
                color = (0, 255, 0) if stress_level < 0.3 else (0, 255, 255) if stress_level < 0.7 else (0, 0, 255)
                cv2.putText(image, f"Stress: {stress_level:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add confidence indicator
            if 'confidence' in indicators:
                confidence = indicators['confidence']
                color = (0, 0, 255) if confidence < 0.5 else (0, 255, 255) if confidence < 0.8 else (0, 255, 0)
                cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            return image
            
        except Exception as e:
            logger.error(f"Error adding visual indicators: {e}")
            return image

class HapticFeedbackEngine:
    """Haptic feedback for accessibility"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize haptic feedback"""
        try:
            # Initialize haptic feedback (platform-specific)
            self.is_initialized = True
            logger.info("Haptic feedback engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing haptic feedback: {e}")
            return False
    
    async def trigger_haptic_feedback(self, feedback_type: str, intensity: str, settings: AccessibilitySettings):
        """Trigger haptic feedback"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Map intensity to vibration patterns
            intensity_map = {
                'light': 1,
                'medium': 2,
                'strong': 3
            }
            
            vibration_intensity = intensity_map.get(intensity, 1)
            
            # Different haptic patterns for different feedback types
            if feedback_type == 'fatigue_warning':
                await self._fatigue_haptic_pattern(vibration_intensity)
            elif feedback_type == 'stress_warning':
                await self._stress_haptic_pattern(vibration_intensity)
            elif feedback_type == 'emergency':
                await self._emergency_haptic_pattern(vibration_intensity)
            elif feedback_type == 'intervention':
                await self._intervention_haptic_pattern(vibration_intensity)
            else:
                await self._default_haptic_pattern(vibration_intensity)
            
        except Exception as e:
            logger.error(f"Error triggering haptic feedback: {e}")
    
    async def _fatigue_haptic_pattern(self, intensity: int):
        """Haptic pattern for fatigue warnings"""
        # Gentle, slow vibration pattern
        for _ in range(intensity):
            # Simulate vibration (would be platform-specific)
            await asyncio.sleep(0.1)
    
    async def _stress_haptic_pattern(self, intensity: int):
        """Haptic pattern for stress warnings"""
        # Quick, sharp vibration pattern
        for _ in range(intensity * 2):
            # Simulate vibration
            await asyncio.sleep(0.05)
    
    async def _emergency_haptic_pattern(self, intensity: int):
        """Haptic pattern for emergency alerts"""
        # Strong, continuous vibration
        for _ in range(intensity * 3):
            # Simulate strong vibration
            await asyncio.sleep(0.2)
    
    async def _intervention_haptic_pattern(self, intensity: int):
        """Haptic pattern for interventions"""
        # Pulsing vibration pattern
        for _ in range(intensity):
            # Simulate pulsing vibration
            await asyncio.sleep(0.3)
    
    async def _default_haptic_pattern(self, intensity: int):
        """Default haptic pattern"""
        # Simple vibration
        for _ in range(intensity):
            await asyncio.sleep(0.1)

class AccessibilityManager:
    """Main accessibility manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tts_engine = TextToSpeechEngine(config.get('tts', {}))
        self.speech_engine = SpeechRecognitionEngine(config.get('speech', {}))
        self.visual_engine = VisualAccessibilityEngine(config.get('visual', {}))
        self.haptic_engine = HapticFeedbackEngine(config.get('haptic', {}))
        
        # User accessibility settings
        self.user_settings = {}
        
    async def initialize(self) -> bool:
        """Initialize all accessibility engines"""
        try:
            # Initialize all engines
            tts_init = await self.tts_engine.initialize()
            speech_init = await self.speech_engine.initialize()
            visual_init = await self.visual_engine.initialize()
            haptic_init = await self.haptic_engine.initialize()
            
            success = all([tts_init, speech_init, visual_init, haptic_init])
            
            if success:
                logger.info("All accessibility engines initialized successfully")
            else:
                logger.warning("Some accessibility engines failed to initialize")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing accessibility manager: {e}")
            return False
    
    def set_user_accessibility_settings(self, user_id: str, settings: AccessibilitySettings):
        """Set accessibility settings for a user"""
        self.user_settings[user_id] = settings
        logger.info(f"Accessibility settings updated for user {user_id}")
    
    def get_user_accessibility_settings(self, user_id: str) -> Optional[AccessibilitySettings]:
        """Get accessibility settings for a user"""
        return self.user_settings.get(user_id)
    
    async def provide_accessibility_feedback(self, user_id: str, feedback_type: str, 
                                            data: Dict[str, Any], message: str = ""):
        """Provide accessibility feedback based on user settings"""
        try:
            settings = self.get_user_accessibility_settings(user_id)
            if not settings:
                logger.warning(f"No accessibility settings found for user {user_id}")
                return
            
            # Audio feedback
            if settings.audio_feedback and message:
                await self.tts_engine.speak(message, settings)
            
            # Haptic feedback
            if settings.haptic_feedback:
                intensity = settings.haptic_intensity
                await self.haptic_engine.trigger_haptic_feedback(feedback_type, intensity, settings)
            
            # Visual feedback (for users with hearing impairments)
            if settings.accessibility_mode == AccessibilityMode.HEARING_IMPAIRMENT:
                await self._provide_visual_feedback(data, settings)
            
        except Exception as e:
            logger.error(f"Error providing accessibility feedback: {e}")
    
    async def _provide_visual_feedback(self, data: Dict[str, Any], settings: AccessibilitySettings):
        """Provide visual feedback for users with hearing impairments"""
        try:
            # Create visual indicators
            if 'image' in data:
                enhanced_image = self.visual_engine.enhance_image_for_visual_impairment(
                    data['image'], settings
                )
                
                # Add visual indicators
                enhanced_image = self.visual_engine.add_visual_indicators(enhanced_image, data)
                
                # Update the image in data
                data['image'] = enhanced_image
            
        except Exception as e:
            logger.error(f"Error providing visual feedback: {e}")
    
    async def handle_voice_command(self, user_id: str, command: str) -> Dict[str, Any]:
        """Handle voice commands for accessibility"""
        try:
            settings = self.get_user_accessibility_settings(user_id)
            if not settings or not settings.voice_commands:
                return {'error': 'Voice commands not enabled'}
            
            # Parse the command
            action, parameters = self.speech_engine.parse_voice_command(command)
            
            # Handle the command
            if action == 'emergency':
                return await self._handle_emergency_command(user_id, parameters)
            elif action == 'fatigue':
                return await self._handle_fatigue_command(user_id, parameters)
            elif action == 'stress':
                return await self._handle_stress_command(user_id, parameters)
            elif action == 'navigation':
                return await self._handle_navigation_command(user_id, parameters)
            elif action == 'settings':
                return await self._handle_settings_command(user_id, parameters)
            elif action == 'status':
                return await self._handle_status_command(user_id, parameters)
            else:
                return {'error': f'Unknown command: {action}'}
            
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
            return {'error': str(e)}
    
    async def _handle_emergency_command(self, user_id: str, parameters: str) -> Dict[str, Any]:
        """Handle emergency voice commands"""
        return {
            'action': 'emergency',
            'message': 'Emergency command received. Alerting emergency contacts.',
            'parameters': parameters
        }
    
    async def _handle_fatigue_command(self, user_id: str, parameters: str) -> Dict[str, Any]:
        """Handle fatigue-related voice commands"""
        return {
            'action': 'fatigue',
            'message': 'Fatigue command received. Monitoring fatigue levels.',
            'parameters': parameters
        }
    
    async def _handle_stress_command(self, user_id: str, parameters: str) -> Dict[str, Any]:
        """Handle stress-related voice commands"""
        return {
            'action': 'stress',
            'message': 'Stress command received. Monitoring stress levels.',
            'parameters': parameters
        }
    
    async def _handle_navigation_command(self, user_id: str, parameters: str) -> Dict[str, Any]:
        """Handle navigation voice commands"""
        return {
            'action': 'navigation',
            'message': 'Navigation command received. Providing directions.',
            'parameters': parameters
        }
    
    async def _handle_settings_command(self, user_id: str, parameters: str) -> Dict[str, Any]:
        """Handle settings voice commands"""
        return {
            'action': 'settings',
            'message': 'Settings command received. Opening settings menu.',
            'parameters': parameters
        }
    
    async def _handle_status_command(self, user_id: str, parameters: str) -> Dict[str, Any]:
        """Handle status voice commands"""
        return {
            'action': 'status',
            'message': 'Status command received. Providing current status.',
            'parameters': parameters
        }
    
    async def listen_for_voice_commands(self, user_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Listen for voice commands from user"""
        try:
            settings = self.get_user_accessibility_settings(user_id)
            if not settings or not settings.voice_commands:
                return None
            
            # Listen for speech
            command_text = await self.speech_engine.listen_for_commands(timeout)
            
            if command_text:
                # Handle the command
                return await self.handle_voice_command(user_id, command_text)
            
            return None
            
        except Exception as e:
            logger.error(f"Error listening for voice commands: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup accessibility manager"""
        try:
            # Cleanup all engines
            await self.tts_engine.cleanup() if hasattr(self.tts_engine, 'cleanup') else None
            await self.speech_engine.cleanup() if hasattr(self.speech_engine, 'cleanup') else None
            await self.visual_engine.cleanup() if hasattr(self.visual_engine, 'cleanup') else None
            await self.haptic_engine.cleanup() if hasattr(self.haptic_engine, 'cleanup') else None
            
            logger.info("Accessibility manager cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during accessibility cleanup: {e}")
