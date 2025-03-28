# modules/stt_manager.py

import time
import warnings
# Make sure Any is imported here
from typing import Optional, Tuple, Any # <--- MODIFIED LINE

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

# Use logger, config manager, and performance monitor
try:
    from .log_manager import logger
    from .config_manager import get_setting
    # Assuming PerformanceMonitor might be passed in or accessed globally/via context
    # For simplicity here, we might instantiate or expect it if needed frequently
    from .performance_monitor import PerformanceMonitor # Or get instance
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log/config/perf managers. Using defaults.")
    # Mock get_setting
    def get_setting(key_path: str, default: Optional[Any] = None) -> Any:
        defaults = {
            "stt.model_size": "tiny.en",
            "stt.device": "cpu",
            "stt.compute_type": "int8",
            "stt.language": None,
            "stt.beam_size": 5,
        }
        return defaults.get(key_path, default)
    # Mock PerformanceMonitor if needed for standalone testing
    class MockPerformanceMonitor:
        def start_timer(self, name): pass
        def stop_timer(self, name, record_count=True): pass
        def record_event(self, name, count=1.0): pass
    performance_monitor = MockPerformanceMonitor() # Example instantiation


# --- Constants ---
EXPECTED_SAMPLE_RATE = 16000 # Whisper models expect 16kHz audio

class STTError(Exception):
    """Custom exception for STT related errors."""
    pass


class STTManager:
    """Handles Speech-to-Text conversion using Faster Whisper."""

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None): # <-- Also updated to use actual class if imported
        """
        Initializes the STTManager by loading the Faster Whisper model.

        Args:
            performance_monitor: An instance of PerformanceMonitor (optional).
        """
        logger.info("Initializing STTManager...")
        # Use actual PerformanceMonitor type if available
        self._perf_monitor: Optional[PerformanceMonitor] = performance_monitor

        self._model_size: str = get_setting("stt.model_size", "base.en")
        self._device: str = get_setting("stt.device", "cpu")
        self._compute_type: str = get_setting("stt.compute_type", "int8")
        self._language: Optional[str] = get_setting("stt.language", None)
        self._beam_size: int = get_setting("stt.beam_size", 5)
        # Add other faster-whisper options from config if needed (e.g., VAD filter)
        # self._vad_filter: bool = get_setting("stt.vad_filter", False)
        # self._vad_parameters: dict = get_setting("stt.vad_parameters", {})

        self._model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self):
        """Loads the Faster Whisper model based on configuration."""
        logger.info(
            "Loading Faster Whisper model: %s (Device: %s, Compute: %s)",
            self._model_size, self._device, self._compute_type
        )
        if self._perf_monitor:
            self._perf_monitor.start_timer("stt_model_load_time")

        # Suppress specific warnings from CTranslate2 or dependencies if needed
        # warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.functional')

        try:
            self._model = WhisperModel(
                model_size_or_path=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                # Pass other options directly if needed:
                # download_root=None,
                # local_files_only=False,
                # num_workers=1, # For multi-GPU, adjust if necessary
                # cpu_threads=0 # 0 for auto
            )
            logger.info("Faster Whisper model loaded successfully.")
            if self._perf_monitor:
                 self._perf_monitor.stop_timer("stt_model_load_time", record_count=False)
                 self._perf_monitor.set_value("stt_model", f"{self._model_size} ({self._compute_type})")

        except ImportError as e:
             logger.error("ImportError loading model. Ensure ctranslate2 and necessary CUDA libs are installed correctly: %s", e, exc_info=True)
             raise STTError(f"Failed to load STT model due to missing dependency: {e}") from e
        except RuntimeError as e:
            logger.error("RuntimeError loading model. Check CUDA/cuDNN compatibility or model files: %s", e, exc_info=True)
            raise STTError(f"Failed to load STT model: {e}") from e
        except Exception as e:
            logger.error("Unexpected error loading STT model: %s", e, exc_info=True)
            if self._perf_monitor:
                # Ensure timer is stopped even on error, don't record count
                try:
                    self._perf_monitor.stop_timer("stt_model_load_time", record_count=False)
                except Exception as timer_e: # Avoid masking original error
                    logger.error("Error stopping performance timer during STT load error: %s", timer_e)
            raise STTError(f"Unexpected error loading STT model: {e}") from e


    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[str, Optional[str], Optional[float]]:
        """
        Transcribes the given audio data to text.

        Args:
            audio_data: NumPy array containing audio data (float32, mono).
            sample_rate: The sample rate of the audio data.

        Returns:
            A tuple containing:
            - The transcribed text (str).
            - The detected language code (str, optional).
            - The language detection probability (float, optional).
            Returns ("", None, None) if transcription fails or no speech is detected.

        Raises:
            STTError: If the STT model is not loaded or transcription fails unexpectedly.
        """
        if self._model is None:
            logger.error("STT model is not loaded. Cannot transcribe.")
            raise STTError("STT model not loaded.")

        if audio_data is None or audio_data.size == 0:
            logger.warning("Attempted to transcribe empty audio data.")
            return "", None, None

        if audio_data.dtype != np.float32:
            logger.warning("STT received audio data that is not float32 (%s). Attempting conversion.", audio_data.dtype)
            try:
                # Ensure conversion respects potential integer range if input was int
                if np.issubdtype(audio_data.dtype, np.integer):
                    max_val = np.iinfo(audio_data.dtype).max
                    audio_data = audio_data.astype(np.float32) / max_val
                else: # Assume it's some other float type or bool maybe?
                    audio_data = audio_data.astype(np.float32)
            except Exception as e:
                logger.error("Failed to convert audio data to float32 for STT: %s", e)
                if self._perf_monitor: self._perf_monitor.record_event("errors", 1)
                return "", None, None # Indicate failure

        if sample_rate != EXPECTED_SAMPLE_RATE:
            # This shouldn't happen if AudioManager resamples correctly, but good to check.
            logger.warning(
                "STT received audio with sample rate %d Hz, but expected %d Hz. "
                "Transcription quality may be affected. Consider resampling in AudioManager.",
                sample_rate, EXPECTED_SAMPLE_RATE
            )
            # NOTE: faster-whisper *might* handle resampling internally via PyAV,
            # but relying on the correct input rate is safer.

        logger.info("Starting audio transcription...")
        if self._perf_monitor:
            self._perf_monitor.start_timer("stt_transcription_time")
            self._perf_monitor.record_event("stt_requests")

        full_text = ""
        detected_language = None
        lang_probability = None

        try:
            # Faster Whisper's transcribe method takes the audio directly
            segments, info = self._model.transcribe(
                audio=audio_data, # Pass the NumPy array
                language=self._language, # Use configured language or None for auto-detect
                beam_size=self._beam_size,
                # Add other relevant options from config:
                # initial_prompt=None,
                # word_timestamps=False, # Set to True if needed later
                # vad_filter=self._vad_filter,
                # vad_parameters=self._vad_parameters,
                # task="transcribe" # or "translate"
            )

            # Process the generator to get the full text
            segment_list = []
            # Use a loop instead of list comprehension for clearer segment logging
            start_time_segments = time.monotonic()
            segment_count = 0
            for segment in segments:
                segment_list.append(segment.text)
                segment_count += 1
                # Log segments as they arrive at DEBUG level if needed
                logger.debug("[STT Segment %d] %.2fs -> %.2fs : %s",
                             segment_count, segment.start, segment.end, segment.text)

            # Check if any segments were produced
            if not segment_list:
                 logger.info("Transcription yielded no segments (likely silence or non-speech).")
                 full_text = ""
                 # info object might still be useful, e.g., for duration
                 detected_language = info.language
                 lang_probability = info.language_probability
                 duration_sec = info.duration
            else:
                 full_text = "".join(segment_list).strip()
                 detected_language = info.language
                 lang_probability = info.language_probability
                 duration_sec = info.duration # Total audio duration processed by whisper


            logger.info(
                "Transcription complete (Audio duration: %.2fs, Detected Lang: %s, Prob: %.2f)",
                duration_sec, detected_language, lang_probability
            )
            # Log full text at INFO, maybe truncate if very long for clarity
            log_text = (full_text[:100] + '...') if len(full_text) > 100 else full_text
            logger.info("Transcription Result: \"%s\"", log_text if log_text else "[No speech detected]")


        except Exception as e:
            logger.error("Error during audio transcription: %s", e, exc_info=True)
            if self._perf_monitor: self._perf_monitor.record_event("errors", 1)
            # Return empty text on error, but could raise STTError if preferred
            return "", None, None
        finally:
            if self._perf_monitor:
                 try:
                    self._perf_monitor.stop_timer("stt_transcription_time") # Always stop timer
                 except Exception as timer_e:
                     logger.error("Error stopping performance timer after STT transcription: %s", timer_e)

        return full_text, detected_language, lang_probability