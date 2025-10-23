# Phase 1: MVP Prototype

## Overview
This phase implements a basic VigilAI prototype using Raspberry Pi 5 with:
- USB webcam for video analysis
- OBD-II adapter for steering data
- Mock wearables integration
- Simple intervention system

## Hardware Requirements
- Raspberry Pi 5 (8GB RAM recommended)
- USB webcam (IR-capable for night vision)
- ELM327 OBD-II dongle
- MicroSD card (32GB+)
- Power supply and cables

## Software Stack
- Python 3.11
- OpenCV 4.8 for video processing
- MediaPipe for facial landmark detection
- SciPy for signal analysis
- FastAPI for web interface

## Key Features
- Real-time video processing (30 FPS)
- Basic facial landmark detection
- Steering data collection via OBD-II
- Simple fatigue scoring algorithm
- Audio/haptic intervention simulation
- Data logging for analysis

## Performance Targets
- <100ms response time
- >95% accuracy for drowsiness detection
- <5% false positives
- <3s user acknowledgment time

## Getting Started
1. Set up Raspberry Pi with required dependencies
2. Connect hardware components
3. Run the main prototype script
4. Monitor performance metrics
