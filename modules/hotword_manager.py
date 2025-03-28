# modules/hotword_manager.py

import time
import threading
import queue
from typing import Optional, List, Dict, Any

import numpy as np
import sounddevice as sd
import openwakeword

# Use logger, config manager, and performance monitor
try:
    from .log_manager import logger
    from .config_manager import get_setting
    from .performance_monitor import PerformanceMonitor
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log/config/perf managers. Using defaults.")
    # Mock get_setting
    def get_setting(key_path: str, default: Optional[Any] = None) -> Any:
        defaults = {
            "hotword.enabled": False,
            "hotword.models": ["hey_jarvis"],
            "hotword.inference_framework": "onnx",
            "hotword.threshold": 0.7,
            "hotword.trigger_level": 1, # OWW internal, usually 1
            "audio.input_device": None,
            # OWW expects 16kHz, chunk size influences latency vs efficiency
            "hotword.chunk_size_ms": 1280 # Default OWW recommendation is often ~1280ms (80*16 samples)
        }
        return defaults.get(key_path, default)
    # Mock PerformanceMonitor
    class MockPerformanceMonitor:
        def record_event(self, name, count=1.0): pass
    # performance_monitor = MockPerformanceMonitor() # Example instantiation


# --- Constants ---
EXPECTED_SAMPLE_RATE = 16000 # openwakeword expects 16kHz

class HotwordError(Exception):
    """Custom exception for Hotword detection errors."""
    pass

class HotwordManager:
    """
    Manages hotword detection using OpenWakeWord in a background thread.
    """

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        logger.info("Initializing HotwordManager...")
        self._perf_monitor = performance_monitor
        self._enabled: bool = get_setting("hotword.enabled", False)
        self._input_device_id: Optional[int] = get_setting("audio.input_device", None)

        self._oww_model: Optional[openwakeword.OpenWakeWord] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._audio_stream: Optional[sd.InputStream] = None
        self._running = threading.Event() # Event to signal thread to stop
        self._detection_lock = threading.Lock()
        self._last_detection: Optional[str] = None # Stores the name of the last detected hotword
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=10) # Queue for audio chunks

        if not self._enabled:
            logger.info("Hotword detection is disabled in configuration.")
            return

        # --- Load Configuration ---
        self._model_names: List[str] = get_setting("hotword.models", ["hey_jarvis"])
        # self._custom_model_paths: List[str] = get_setting("hotword.custom_model_paths", []) # If supporting custom
        self._inference_framework: str = get_setting("hotword.inference_framework", "onnx")
        self._threshold: float = get_setting("hotword.threshold", 0.7)
        # Note: trigger_level is often handled internally by oww based on chunk size, but check docs if issues arise
        # self._trigger_level: int = get_setting("hotword.trigger_level", 1)
        self._chunk_size_ms: int = get_setting("hotword.chunk_size_ms", 1280)

        # Calculate chunk size in samples
        self._chunk_samples = int(EXPECTED_SAMPLE_RATE * self._chunk_size_ms / 1000)

        if not self._model_names:
            logger.warning("Hotword detection enabled, but no models specified in config. Disabling.")
            self._enabled = False
            return

        self._load_models()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_running(self) -> bool:
        return self._listener_thread is not None and self._listener_thread.is_alive()

    def _load_models(self):
        """Loads the OpenWakeWord models."""
        if not self._enabled: return

        logger.info("Loading OpenWakeWord models: %s", self._model_names)
        # Add performance timing if desired
        try:
            # Combine pre-defined and custom paths if feature is added
            # model_paths = self._model_names + self._custom_model_paths
            self._oww_model = openwakeword.OpenWakeWord(
                wakeword_models=self._model_names, # Pass list of names/paths
                inference_framework=self._inference_framework,
                # Other parameters if needed (e.g., custom thresholds per model)
            )
            logger.info("OpenWakeWord models loaded successfully.")
            # Store activation counts per model if needed for trigger level logic
            # self._activation_counts = {name: 0 for name in self._oww_model.models}

        except Exception as e:
            logger.error("Failed to load OpenWakeWord models: %s", e, exc_info=True)
            if self._perf_monitor: self._perf_monitor.record_event("errors", 1)
            self._enabled = False # Disable if loading fails
            raise HotwordError(f"Failed to load OpenWakeWord models: {e}") from e

    def start(self):
        """Starts the background hotword listening thread."""
        if not self._enabled:
            logger.debug("Cannot start HotwordManager: disabled.")
            return
        if self.is_running:
            logger.warning("HotwordManager listener thread already running.")
            return
        if self._oww_model is None:
            logger.error("Cannot start HotwordManager: models not loaded.")
            return

        logger.info("Starting hotword listener thread...")
        self._running.set() # Signal that the thread should run
        self._last_detection = None # Reset detection state
        self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        self._listener_thread.start()

    def stop(self):
        """Stops the background hotword listening thread."""
        if not self.is_running:
            logger.debug("HotwordManager listener thread is not running.")
            return

        logger.info("Stopping hotword listener thread...")
        self._running.clear() # Signal thread to stop

        # Put a sentinel value in the queue to unblock the get() if necessary
        try:
            self._chunk_queue.put_nowait(None)
        except queue.Full:
            pass # Queue might be full, thread should check self._running soon anyway

        if self._listener_thread:
            self._listener_thread.join(timeout=2.0) # Wait for thread to finish
            if self._listener_thread.is_alive():
                logger.warning("Hotword listener thread did not stop gracefully.")
            self._listener_thread = None

        # Clear the queue after stopping
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Hotword listener stopped.")


    def _listener_loop(self):
        """The main loop running in the background thread."""
        try:
            logger.debug("Hotword listener thread started.")
            # Setup audio stream
            self._audio_stream = sd.InputStream(
                samplerate=EXPECTED_SAMPLE_RATE,
                channels=1,
                dtype='int16', # OWW expects int16
                blocksize=self._chunk_samples, # Use calculated chunk size
                device=self._input_device_id,
                callback=self._audio_callback
            )
            self._audio_stream.start()
            logger.info("Hotword audio stream started. Listening...")

            while self._running.is_set():
                try:
                    # Get chunk from the queue filled by the callback
                    chunk = self._chunk_queue.get(timeout=0.5) # Wait briefly
                    if chunk is None: # Sentinel value
                         break

                    # Feed chunk to OpenWakeWord
                    if self._oww_model:
                        prediction = self._oww_model.predict(chunk)

                        # Check results (prediction is a dict: {'model_name': score})
                        for model_name, score in prediction.items():
                            if score >= self._threshold:
                                logger.info(
                                    "Hotword detected: '%s' (Score: %.2f)",
                                    model_name, score
                                )
                                with self._detection_lock:
                                    self._last_detection = model_name
                                if self._perf_monitor:
                                    self._perf_monitor.record_event("hotword_detections", 1)
                                # Optional: Add a brief pause/cooldown after detection?
                                # time.sleep(1.0)
                                # Reset OWW internal state if needed after detection? Check docs.
                                # self._oww_model.reset() # Example if needed

                except queue.Empty:
                    # Timeout waiting for audio chunk, just continue loop if running
                    continue
                except Exception as e:
                    logger.error("Error in hotword listener loop: %s", e, exc_info=True)
                    if self._perf_monitor: self._perf_monitor.record_event("errors", 1)
                    time.sleep(1) # Avoid spamming logs on continuous errors

        except sd.PortAudioError as e:
            logger.error("PortAudioError setting up hotword stream: %s", e, exc_info=True)
            if self._perf_monitor: self._perf_monitor.record_event("errors", 1)
            self._enabled = False # Disable if stream fails
        except Exception as e:
            logger.error("Unexpected error setting up hotword stream: %s", e, exc_info=True)
            if self._perf_monitor: self._perf_monitor.record_event("errors", 1)
            self._enabled = False
        finally:
            # Cleanup stream
            if self._audio_stream:
                try:
                    if not self._audio_stream.closed:
                        self._audio_stream.stop()
                        self._audio_stream.close()
                    logger.debug("Hotword audio stream closed.")
                except Exception as e:
                    logger.error("Error closing hotword audio stream: %s", e)
            self._audio_stream = None
            logger.debug("Hotword listener thread finished.")


    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
        """Callback function for the sounddevice InputStream."""
        if status:
            logger.warning("Hotword InputStream status: %s", status)
            if self._perf_monitor: self._perf_monitor.record_event("audio_input_errors", 1)
            return

        if not self._running.is_set():
            return # Don't process if stopping

        try:
            # indata should already be int16 based on stream setup
            self._chunk_queue.put_nowait(indata)
        except queue.Full:
            logger.warning("Hotword audio queue is full. Dropping chunk.")
            if self._perf_monitor: self._perf_monitor.record_event("audio_drops", 1)


    def get_detected_keyword(self) -> Optional[str]:
        """
        Checks if a hotword has been detected since the last call.

        This method is thread-safe and resets the detection flag.

        Returns:
            The name of the detected keyword (str) or None if no detection occurred.
        """
        if not self._enabled:
            return None

        detected = None
        with self._detection_lock:
            detected = self._last_detection
            self._last_detection = None # Reset after checking
        return detected