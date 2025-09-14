#!/usr/bin/env python3
"""
Dual-Device ML Controller for Minecraft
=======================================

Runs two ML models simultaneously using threading:
1. EEG Model: Controls forward movement (W) and jumping (SPACE)
2. EMG Model: Controls camera (arrows), mining (left click), placing (right click)

Usage: python main.py --eeg-model ../eeg/eeg_model.pkl --emg-model ../mindrove/gesture_model.pkl
"""

import argparse
import threading
import time
import queue
import logging
from typing import Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ControlCommand:
    """Unified control command from either EEG or EMG"""
    device: str  # "eeg" or "emg"
    action: str  # "forward", "jump", "mine", "place", "look_left", "look_right", "idle"
    confidence: float
    timestamp: float

class DualDeviceController:
    """Main controller that coordinates EEG and EMG models"""
    
    def __init__(self, eeg_model_path: str, emg_model_path: str):
        self.eeg_model_path = eeg_model_path
        self.emg_model_path = emg_model_path
        
        # Thread-safe command queue
        self.command_queue = queue.Queue(maxsize=100)
        
        # Control threads
        self.eeg_thread: Optional[threading.Thread] = None
        self.emg_thread: Optional[threading.Thread] = None
        self.executor_thread: Optional[threading.Thread] = None
        
        # Thread control
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Import controllers (lazy loading to avoid import errors)
        self.eeg_controller = None
        self.emg_controller = None
        
    def start(self):
        """Start both ML model threads and command executor"""
        logger.info("🚀 Starting Dual-Device ML Controller")
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start EEG controller thread
        self.eeg_thread = threading.Thread(
            target=self._run_eeg_controller,
            name="EEG-Controller",
            daemon=True
        )
        self.eeg_thread.start()
        logger.info("✅ EEG controller thread started")
        
        # Start EMG controller thread  
        self.emg_thread = threading.Thread(
            target=self._run_emg_controller,
            name="EMG-Controller", 
            daemon=True
        )
        self.emg_thread.start()
        logger.info("✅ EMG controller thread started")
        
        # Start command executor thread
        self.executor_thread = threading.Thread(
            target=self._run_command_executor,
            name="Command-Executor",
            daemon=True
        )
        self.executor_thread.start()
        logger.info("✅ Command executor thread started")
        
        logger.info("🎮 All systems ready! Minecraft control active.")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
            self.stop()
    
    def stop(self):
        """Gracefully shutdown all threads"""
        logger.info("🔄 Shutting down controllers...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Wait for threads to finish
        if self.eeg_thread and self.eeg_thread.is_alive():
            self.eeg_thread.join(timeout=2.0)
            
        if self.emg_thread and self.emg_thread.is_alive():
            self.emg_thread.join(timeout=2.0)
            
        if self.executor_thread and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=2.0)
        
        logger.info("✅ All controllers stopped")
    
    def _run_eeg_controller(self):
        """Run EEG model in dedicated thread"""
        try:
            # Import EEG controller
            from Data_Collection.eeg_key_mapper import EEGController
            
            logger.info("🧠 Initializing EEG controller...")
            self.eeg_controller = EEGController(
                model_path=self.eeg_model_path,
                command_queue=self.command_queue
            )
            
            logger.info("🧠 EEG controller ready - monitoring for forward/jump")
            self.eeg_controller.run(self.shutdown_event)
            
        except Exception as e:
            logger.error(f"❌ EEG controller error: {e}")
            self.running = False
    
    def _run_emg_controller(self):
        """Run EMG model in dedicated thread"""
        try:
            # Import EMG controller
            from minecraft_cli.ml_key_mapper import MLBurstDetector, MinecraftController
            from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
            import pickle
            
            logger.info("💪 Initializing EMG controller...")
            
            # Load EMG model
            with open(self.emg_model_path, 'rb') as f:
                emg_model = pickle.load(f)
            
            # Setup MindRove board
            params = MindRoveInputParams()
            params.ip_address = "192.168.4.1"
            params.ip_port = 4210
            
            board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD.value, params)
            board.prepare_session()
            board.start_stream()
            
            # Initialize detectors
            burst_detector = MLBurstDetector(emg_model)
            controller = MinecraftController(command_queue=self.command_queue)
            
            logger.info("💪 EMG controller ready - monitoring for gestures")
            
            # Main EMG detection loop
            while not self.shutdown_event.is_set():
                try:
                    data = board.get_board_data()
                    if data.shape[1] > 0:
                        gesture, confidence = burst_detector.process_data(data)
                        if gesture != 'idle' and confidence > 0.7:
                            command = ControlCommand(
                                device="emg",
                                action=gesture,
                                confidence=confidence,
                                timestamp=time.time()
                            )
                            try:
                                self.command_queue.put_nowait(command)
                            except queue.Full:
                                pass  # Skip if queue full
                    
                    time.sleep(0.02)  # 50Hz processing
                    
                except Exception as e:
                    logger.warning(f"EMG processing error: {e}")
                    time.sleep(0.1)
            
            # Cleanup
            board.stop_stream()
            board.release_session()
            
        except Exception as e:
            logger.error(f"❌ EMG controller error: {e}")
            self.running = False
    
    def _run_command_executor(self):
        """Execute commands from both controllers with priority handling"""
        try:
            import pynput
            from pynput import mouse, keyboard as kb
            
            mouse_controller = mouse.Controller()
            keyboard_controller = kb.Controller()
            
            logger.info("⚡ Command executor ready")
            
            # State tracking
            current_forward = False
            last_command_time = {}
            
            while not self.shutdown_event.is_set():
                try:
                    # Get command with timeout
                    command = self.command_queue.get(timeout=0.1)
                    
                    # Prevent command spam (rate limiting)
                    current_time = time.time()
                    action_key = f"{command.device}_{command.action}"
                    
                    if action_key in last_command_time:
                        if current_time - last_command_time[action_key] < 0.2:  # 200ms cooldown
                            continue
                    
                    last_command_time[action_key] = current_time
                    
                    # Execute command based on device and action
                    if command.device == "eeg":
                        self._execute_eeg_command(command, keyboard_controller, current_forward)
                    elif command.device == "emg":
                        self._execute_emg_command(command, mouse_controller, keyboard_controller)
                    
                    logger.debug(f"🎯 Executed {command.device} → {command.action} (conf: {command.confidence:.2f})")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.warning(f"Command execution error: {e}")
        
        except Exception as e:
            logger.error(f"❌ Command executor error: {e}")
    
    def _execute_eeg_command(self, command: ControlCommand, keyboard_controller, current_forward: bool):
        """Execute EEG commands (forward/jump)"""
        if command.action == "forward":
            if not current_forward:
                keyboard_controller.press('w')
                current_forward = True
                logger.info("🏃 Forward START")
        elif command.action == "stop_forward":
            if current_forward:
                keyboard_controller.release('w')
                current_forward = False
                logger.info("🛑 Forward STOP")
        elif command.action == "jump":
            keyboard_controller.press(' ')
            time.sleep(0.1)
            keyboard_controller.release(' ')
            logger.info("🦘 JUMP")
    
    def _execute_emg_command(self, command: ControlCommand, mouse_controller, keyboard_controller):
        """Execute EMG commands (mine/place/look)"""
        from pynput import mouse, keyboard as kb
        
        if command.action == "swing_down":  # Mine
            mouse_controller.click(mouse.Button.left)
            logger.info("⛏️ MINE")
        elif command.action == "arm_up":  # Place
            mouse_controller.click(mouse.Button.right)
            logger.info("🧱 PLACE")
        elif command.action == "wrist_flex_left":  # Look left
            keyboard_controller.press(kb.Key.left)
            time.sleep(0.1)
            keyboard_controller.release(kb.Key.left)
            logger.info("👈 LOOK LEFT")
        elif command.action == "wrist_supinate_right":  # Look right
            keyboard_controller.press(kb.Key.right)
            time.sleep(0.1)
            keyboard_controller.release(kb.Key.right)
            logger.info("👉 LOOK RIGHT")

def main():
    parser = argparse.ArgumentParser(description="Dual-Device ML Controller for Minecraft")
    parser.add_argument("--eeg-model", required=True, help="Path to trained EEG model (.pkl)")
    parser.add_argument("--emg-model", required=True, help="Path to trained EMG model (.pkl)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start controller
    controller = DualDeviceController(
        eeg_model_path=args.eeg_model,
        emg_model_path=args.emg_model
    )
    
    try:
        controller.start()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()

if __name__ == "__main__":
    main()