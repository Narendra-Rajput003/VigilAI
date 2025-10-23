"""
Intervention System for VigilAI
Handles non-distracting interventions for driver fatigue and stress
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional
import json

# Optional pygame import for audio
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

logger = logging.getLogger(__name__)

class InterventionSystem:
    """Manages intervention strategies for driver fatigue and stress"""
    
    def __init__(self, config):
        self.config = config
        
        # Intervention settings
        self.intervention_types = config.get("intervention_types", ["audio", "haptic", "visual"])
        self.escalation_levels = config.get("escalation_levels", 3)
        self.cooldown_period = config.get("cooldown_period", 30)  # seconds
        
        # Audio settings
        self.audio_enabled = config.get("audio_enabled", True)
        self.audio_volume = config.get("audio_volume", 0.7)
        
        # Haptic settings
        self.haptic_enabled = config.get("haptic_enabled", True)
        self.haptic_intensity = config.get("haptic_intensity", 0.5)
        
        # Visual settings
        self.visual_enabled = config.get("visual_enabled", True)
        
        # Intervention state
        self.active_interventions: List[Dict] = []
        self.intervention_history: List[Dict] = []
        self.last_intervention_time = 0
        
        # Initialize audio system
        if self.audio_enabled:
            self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize audio system for interventions"""
        if not PYGAME_AVAILABLE:
            logger.warning("Pygame not available, audio interventions disabled")
            self.audio_enabled = False
            return
        
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            logger.info("Audio system initialized")
        except Exception as e:
            logger.error(f"Error initializing audio: {e}")
            self.audio_enabled = False
    
    async def trigger_intervention(self, intervention_type: str, severity: float) -> bool:
        """Trigger an intervention based on detection results"""
        try:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_intervention_time < self.cooldown_period:
                logger.debug("Intervention in cooldown period")
                return False
            
            # Determine intervention level based on severity
            level = self._calculate_intervention_level(severity)
            
            # Create intervention
            intervention = {
                "type": intervention_type,
                "level": level,
                "severity": severity,
                "timestamp": current_time,
                "id": len(self.intervention_history)
            }
            
            # Execute intervention
            success = await self._execute_intervention(intervention)
            
            if success:
                # Record intervention
                self.active_interventions.append(intervention)
                self.intervention_history.append(intervention)
                self.last_intervention_time = current_time
                
                logger.info(f"Intervention triggered: {intervention_type} level {level} "
                          f"(severity: {severity:.2f})")
                
                # Schedule intervention cleanup
                asyncio.create_task(self._cleanup_intervention(intervention))
            
            return success
            
        except Exception as e:
            logger.error(f"Error triggering intervention: {e}")
            return False
    
    def _calculate_intervention_level(self, severity: float) -> int:
        """Calculate intervention level based on severity (0.0 to 1.0)"""
        if severity < 0.3:
            return 1  # Mild intervention
        elif severity < 0.7:
            return 2  # Moderate intervention
        else:
            return 3  # Strong intervention
    
    async def _execute_intervention(self, intervention: Dict) -> bool:
        """Execute the specified intervention"""
        intervention_type = intervention["type"]
        level = intervention["level"]
        
        try:
            if intervention_type == "fatigue":
                return await self._execute_fatigue_intervention(level)
            elif intervention_type == "stress":
                return await self._execute_stress_intervention(level)
            else:
                logger.warning(f"Unknown intervention type: {intervention_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing intervention: {e}")
            return False
    
    async def _execute_fatigue_intervention(self, level: int) -> bool:
        """Execute fatigue-specific interventions"""
        success = True
        
        # Audio intervention
        if self.audio_enabled:
            audio_success = await self._play_fatigue_audio(level)
            success = success and audio_success
        
        # Haptic intervention
        if self.haptic_enabled:
            haptic_success = await self._trigger_fatigue_haptic(level)
            success = success and haptic_success
        
        # Visual intervention
        if self.visual_enabled:
            visual_success = await self._show_fatigue_visual(level)
            success = success and visual_success
        
        return success
    
    async def _execute_stress_intervention(self, level: int) -> bool:
        """Execute stress-specific interventions"""
        success = True
        
        # Audio intervention
        if self.audio_enabled:
            audio_success = await self._play_stress_audio(level)
            success = success and audio_success
        
        # Haptic intervention
        if self.haptic_enabled:
            haptic_success = await self._trigger_stress_haptic(level)
            success = success and haptic_success
        
        # Visual intervention
        if self.visual_enabled:
            visual_success = await self._show_stress_visual(level)
            success = success and visual_success
        
        return success
    
    async def _play_fatigue_audio(self, level: int) -> bool:
        """Play audio intervention for fatigue"""
        try:
            if level == 1:
                message = "Gentle reminder: Take a break if you feel tired."
                tone = "gentle"
            elif level == 2:
                message = "Warning: Signs of fatigue detected. Please pull over safely."
                tone = "warning"
            else:
                message = "URGENT: Pull over immediately. You appear to be very tired."
                tone = "urgent"
            
            # In a real implementation, you would use TTS
            logger.info(f"Audio intervention (level {level}): {message}")
            
            # Simulate audio playback
            await asyncio.sleep(2.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing fatigue audio: {e}")
            return False
    
    async def _play_stress_audio(self, level: int) -> bool:
        """Play audio intervention for stress"""
        try:
            if level == 1:
                message = "Relaxation reminder: Take deep breaths and stay calm."
                tone = "calming"
            elif level == 2:
                message = "Stress detected: Try to relax and focus on driving safely."
                tone = "concerned"
            else:
                message = "High stress levels detected. Consider pulling over to calm down."
                tone = "urgent"
            
            logger.info(f"Audio intervention (level {level}): {message}")
            
            # Simulate audio playback
            await asyncio.sleep(2.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing stress audio: {e}")
            return False
    
    async def _trigger_fatigue_haptic(self, level: int) -> bool:
        """Trigger haptic intervention for fatigue"""
        try:
            if level == 1:
                pattern = "gentle_vibration"
                duration = 1.0
            elif level == 2:
                pattern = "pulsing_vibration"
                duration = 2.0
            else:
                pattern = "strong_vibration"
                duration = 3.0
            
            logger.info(f"Haptic intervention (level {level}): {pattern} for {duration}s")
            
            # Simulate haptic feedback
            await asyncio.sleep(duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering fatigue haptic: {e}")
            return False
    
    async def _trigger_stress_haptic(self, level: int) -> bool:
        """Trigger haptic intervention for stress"""
        try:
            if level == 1:
                pattern = "calming_vibration"
                duration = 1.5
            elif level == 2:
                pattern = "rhythmic_vibration"
                duration = 2.5
            else:
                pattern = "intense_vibration"
                duration = 3.5
            
            logger.info(f"Haptic intervention (level {level}): {pattern} for {duration}s")
            
            # Simulate haptic feedback
            await asyncio.sleep(duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering stress haptic: {e}")
            return False
    
    async def _show_fatigue_visual(self, level: int) -> bool:
        """Show visual intervention for fatigue"""
        try:
            if level == 1:
                message = "😴 Take a break if you feel tired"
                color = "yellow"
            elif level == 2:
                message = "⚠️ Fatigue detected - Pull over safely"
                color = "orange"
            else:
                message = "🚨 URGENT: Pull over immediately"
                color = "red"
            
            logger.info(f"Visual intervention (level {level}): {message}")
            
            # In a real implementation, you would display this on the dashboard
            # For now, just log the message
            
            return True
            
        except Exception as e:
            logger.error(f"Error showing fatigue visual: {e}")
            return False
    
    async def _show_stress_visual(self, level: int) -> bool:
        """Show visual intervention for stress"""
        try:
            if level == 1:
                message = "🧘 Take deep breaths and stay calm"
                color = "blue"
            elif level == 2:
                message = "😰 Stress detected - Focus on safe driving"
                color = "orange"
            else:
                message = "🚨 High stress - Consider pulling over"
                color = "red"
            
            logger.info(f"Visual intervention (level {level}): {message}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error showing stress visual: {e}")
            return False
    
    async def _cleanup_intervention(self, intervention: Dict):
        """Cleanup intervention after completion"""
        try:
            # Wait for intervention duration
            await asyncio.sleep(5.0)  # 5 second intervention duration
            
            # Remove from active interventions
            if intervention in self.active_interventions:
                self.active_interventions.remove(intervention)
            
            logger.debug(f"Intervention {intervention['id']} completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up intervention: {e}")
    
    def get_active_interventions(self) -> List[Dict]:
        """Get currently active interventions"""
        return self.active_interventions.copy()
    
    def get_intervention_history(self, limit: int = 50) -> List[Dict]:
        """Get intervention history"""
        return self.intervention_history[-limit:]
    
    def get_intervention_statistics(self) -> Dict:
        """Get intervention statistics"""
        if not self.intervention_history:
            return {
                "total_interventions": 0,
                "fatigue_interventions": 0,
                "stress_interventions": 0,
                "avg_severity": 0.0,
                "active_count": 0
            }
        
        fatigue_count = sum(1 for i in self.intervention_history if i["type"] == "fatigue")
        stress_count = sum(1 for i in self.intervention_history if i["type"] == "stress")
        avg_severity = sum(i["severity"] for i in self.intervention_history) / len(self.intervention_history)
        
        return {
            "total_interventions": len(self.intervention_history),
            "fatigue_interventions": fatigue_count,
            "stress_interventions": stress_count,
            "avg_severity": avg_severity,
            "active_count": len(self.active_interventions)
        }
    
    async def stop_all_interventions(self):
        """Stop all active interventions"""
        try:
            # Clear active interventions
            self.active_interventions.clear()
            
            # Stop audio if playing
            if self.audio_enabled and PYGAME_AVAILABLE:
                pygame.mixer.stop()
            
            logger.info("All interventions stopped")
            
        except Exception as e:
            logger.error(f"Error stopping interventions: {e}")
    
    async def cleanup(self):
        """Cleanup intervention system"""
        try:
            # Stop all active interventions
            await self.stop_all_interventions()
            
            # Quit pygame mixer
            if self.audio_enabled and PYGAME_AVAILABLE:
                pygame.mixer.quit()
            
            logger.info("Intervention system cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during intervention cleanup: {e}")
