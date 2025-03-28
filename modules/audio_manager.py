# modules/audio_manager.py

import time
import queue
import threading
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import sounddevice as sd
import webrtcvad
import wave
from pathlib import Path # Added for save_wave directory check
from scipy.signal import resample_poly # For potential resampling if needed

# Use logger and config manager
try:
    from .log_manager import logger
    from .config_manager import get_setting
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log/config managers. Using defaults.")
    # Mock get_setting for standalone testing if needed
    def get_setting(key_path: str, default: Any = None) -> Any:
        defaults = {
            "audio.input_device": None, "audio.output_device": None,
            "audio.vad.enabled": True, "audio.vad.sample_rate": 16000,
            "audio.vad.frame_duration_ms": 30, "audio.vad.aggressiveness": 2,
            "audio.vad.silence_duration_ms": 1200, "audio.vad.min_record_duration_ms": 500,
            "tts.normalize_volume": True, "tts.sample_rate": 24000,
        }
        keys = key_path.split('.')
        val = defaults
        try:
             for k in keys: val = val[k]
             return val
        except (KeyError, TypeError): return default


class AudioError(Exception):
    """Custom exception for audio-related errors."""
    pass


class AudioManager:
    """Handles audio recording (with VAD) and playback."""

    # VAD requires specific sample rates
    _VAD_SUPPORTED_RATES = {8000, 16000, 32000, 48000}

    def __init__(self):
        logger.info("Initializing AudioManager...")
        self._input_device_id: Optional[int] = self._get_device_id("input")
        self._output_device_id: Optional[int] = self._get_device_id("output")

        # VAD specific setup
        self._vad_enabled: bool = get_setting("audio.vad.enabled", True)
        if self._vad_enabled:
            self._vad_sample_rate: int = get_setting("audio.vad.sample_rate", 16000)
            self._vad_frame_duration: int = get_setting("audio.vad.frame_duration_ms", 30)
            self._vad_aggressiveness: int = get_setting("audio.vad.aggressiveness", 2)
            self._vad_silence_ms: int = get_setting("audio.vad.silence_duration_ms", 1200)
            self._vad_min_record_ms: int = get_setting("audio.vad.min_record_duration_ms", 500)

            if self._vad_sample_rate not in self._VAD_SUPPORTED_RATES:
                raise AudioError(f"VAD sample rate {self._vad_sample_rate}Hz not supported.")
            if self._vad_frame_duration not in [10, 20, 30]:
                 raise AudioError("VAD frame duration must be 10, 20, or 30 ms.")
            if not 0 <= self._vad_aggressiveness <= 3:
                raise AudioError("VAD aggressiveness must be between 0 and 3.")

            try:
                self._vad = webrtcvad.Vad(self._vad_aggressiveness)
                logger.info("VAD initialized (Rate: %dHz, Frame: %dms, Silence: %dms, Aggressiveness: %d)",
                            self._vad_sample_rate, self._vad_frame_duration, self._vad_silence_ms, self._vad_aggressiveness)
            except Exception as e:
                logger.error("Failed to initialize WebRTC VAD: %s", e, exc_info=True)
                raise AudioError("Failed to initialize WebRTC VAD") from e
        else:
            self._vad = None
            logger.info("VAD is disabled via configuration.")

        self._normalize_playback: bool = get_setting("tts.normalize_volume", True)
        self._list_devices() # Log available devices for debugging

        # --- State for Async Recording ---
        self._async_recording_thread: Optional[threading.Thread] = None
        self._async_recording_stop_event = threading.Event()
        self._async_audio_buffer: List[np.ndarray] = []
        self._async_recording_active = threading.Lock() # Lock to manage access/state
        self._async_sample_rate: int = 16000 # Default, set during start

    def _get_device_id(self, device_type: str) -> Optional[int]:
        """Gets the configured device ID, handling 'null' for default."""
        device_setting = get_setting(f"audio.{device_type}_device", None)
        if device_setting is None or str(device_setting).lower() == 'null':
            logger.info("Using default %s audio device.", device_type)
            return None
        try:
            devices = sd.query_devices()
            device_id = int(device_setting)
            if 0 <= device_id < len(devices):
                 dev_info = devices[device_id]
                 if device_type == "input" and dev_info.get('max_input_channels', 0) > 0: return device_id
                 elif device_type == "output" and dev_info.get('max_output_channels', 0) > 0: return device_id
                 else:
                     logger.warning("Device ID %d (%s) does not support %s. Using default.", device_id, dev_info.get('name'), device_type)
                     return None
            else:
                 logger.warning("Invalid audio device ID '%s'. Must be between 0 and %d. Using default.", device_setting, len(devices) -1)
                 return None
        except (ValueError, TypeError):
            logger.warning("Invalid audio device ID format '%s' for %s device. Using default.", device_setting, device_type)
            return None
        except sd.PortAudioError as e:
             logger.error("PortAudio error querying devices: %s. Using default devices.", e)
             return None
        except Exception as e:
             logger.error("Unexpected error getting device ID: %s. Using default.", e)
             return None

    def _list_devices(self):
        """Logs available audio devices and checks settings."""
        try:
            devices = sd.query_devices()
            logger.debug("Available audio devices:\n%s", devices)

            default_input_info = None; default_output_info = None
            try: default_input_info = sd.query_devices(kind='input')
            except Exception as e_in: logger.warning("Error querying default input device: %s", e_in)
            try: default_output_info = sd.query_devices(kind='output')
            except Exception as e_out: logger.warning("Error querying default output device: %s", e_out)

            input_dev_idx: Any = 'Default'; input_dev_name = "Default (Not Found)"
            if self._input_device_id is not None:
                input_dev_idx = self._input_device_id
                if 0 <= input_dev_idx < len(devices): input_dev_name = devices[input_dev_idx].get('name', 'Unknown')
                else: input_dev_name = f"Invalid Index ({input_dev_idx})"
            elif default_input_info:
                input_dev_idx = default_input_info.get('index', 'Default')
                input_dev_name = default_input_info.get('name', 'Default Input')

            output_dev_idx: Any = 'Default'; output_dev_name = "Default (Not Found)"
            if self._output_device_id is not None:
                output_dev_idx = self._output_device_id
                if 0 <= output_dev_idx < len(devices): output_dev_name = devices[output_dev_idx].get('name', 'Unknown')
                else: output_dev_name = f"Invalid Index ({output_dev_idx})"
            elif default_output_info:
                output_dev_idx = default_output_info.get('index', 'Default')
                output_dev_name = default_output_info.get('name', 'Default Output')

            logger.info("Selected Input Device: %s - %s", str(input_dev_idx), input_dev_name)
            logger.info("Selected Output Device: %s - %s", str(output_dev_idx), output_dev_name)

            vad_rate = self._vad_sample_rate if self._vad_enabled else 16000
            try:
                 sd.check_input_settings(device=self._input_device_id, channels=1, samplerate=vad_rate)
                 logger.debug("Input device settings check passed (Rate: %d Hz).", vad_rate)
            except (ValueError, sd.PortAudioError) as e:
                 logger.warning("Input device settings check failed for device %s: %s", str(input_dev_idx), e)

            tts_rate = get_setting("tts.sample_rate", 24000)
            try:
                 sd.check_output_settings(device=self._output_device_id, channels=1, samplerate=tts_rate)
                 logger.debug("Output device settings check passed (Rate: %d Hz).", tts_rate)
            except (ValueError, sd.PortAudioError) as e:
                 logger.warning("Output device settings check failed for device %s: %s", str(output_dev_idx), e)

        except sd.PortAudioError as e: logger.error("PortAudio error during device listing/checking: %s", e)
        except Exception as e: logger.error("Error listing or checking audio devices: %s", e, exc_info=True)

    def record_audio(
        self,
        target_sample_rate: int,
        duration_seconds: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Records audio. Uses VAD if enabled and duration is None.
        Uses fixed duration if duration_seconds is provided.
        For non-VAD PTT, use start/stop_async_recording methods.
        """
        if duration_seconds is not None:
            record_sample_rate = target_sample_rate
            num_frames = int(duration_seconds * record_sample_rate)
            logger.info("Starting fixed duration recording: %.2f seconds at %d Hz...", duration_seconds, record_sample_rate)
            try:
                audio_data = sd.rec(frames=num_frames, samplerate=record_sample_rate, channels=1, dtype='float32', device=self._input_device_id)
                sd.wait()
                logger.info("Fixed duration recording finished.")
                return audio_data.flatten() if audio_data.size > 0 else None
            except sd.PortAudioError as e: raise AudioError(f"Audio recording failed: {e}") from e
            except Exception as e: raise AudioError(f"Unexpected recording error: {e}") from e

        elif self._vad_enabled and self._vad:
            logger.info("Starting VAD recording (target rate: %d Hz, VAD rate: %d Hz)", target_sample_rate, self._vad_sample_rate)
            audio_data_vad_rate = self._record_with_vad()
            if audio_data_vad_rate is None or audio_data_vad_rate.size == 0: logger.warning("VAD recording captured no audio."); return None
            if self._vad_sample_rate != target_sample_rate:
                logger.debug("Resampling VAD audio from %d Hz to %d Hz", self._vad_sample_rate, target_sample_rate)
                try:
                    audio_data = resample_poly(audio_data_vad_rate, target_sample_rate, self._vad_sample_rate).astype(np.float32)
                    logger.debug("Resampling complete. New length: %d samples", len(audio_data))
                    return audio_data.flatten()
                except Exception as e: raise AudioError("Failed to resample recorded audio") from e
            else:
                 return audio_data_vad_rate.flatten()
        else:
            raise AudioError("Cannot record: Specify duration_seconds or use start/stop_async_recording for non-VAD PTT.")

    def _record_with_vad(self) -> Optional[np.ndarray]:
        """Internal helper for VAD-based recording. Returns float32 at VAD rate."""
        frames_per_buffer = int(self._vad_sample_rate * self._vad_frame_duration / 1000)
        vad_bytes_per_frame = frames_per_buffer * 2
        silence_frames_needed = int(self._vad_silence_ms / self._vad_frame_duration)
        min_record_frames = int(self._vad_min_record_ms / self._vad_frame_duration)
        recorded_frames_bytes: List[bytes] = []
        consecutive_silence_frames = 0; triggered = False; total_frames = 0
        start_time = time.monotonic()
        audio_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=50)

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
            if status: logger.warning("Sounddevice callback status: %s", str(status))
            try:
                if isinstance(indata, np.ndarray) and indata.dtype == np.int16: audio_queue.put_nowait(indata.tobytes())
                elif isinstance(indata, np.ndarray): logger.error("Callback wrong dtype: %s", indata.dtype)
                else: logger.error("Callback non-numpy data: %s", type(indata))
            except queue.Full: logger.warning("Audio queue full in VAD callback.")
            except Exception as cb_e: logger.error("Error in VAD audio callback: %s", cb_e)

        logger.info("Listening... (Silence threshold: %d frames = %d ms)", silence_frames_needed, self._vad_silence_ms)
        stream: Optional[sd.InputStream] = None
        try:
            stream = sd.InputStream(samplerate=self._vad_sample_rate, channels=1, dtype='int16', blocksize=frames_per_buffer, device=self._input_device_id, callback=audio_callback)
            stream.start()
            last_vad_check_time = time.monotonic()
            while True:
                now = time.monotonic()
                try: frame_bytes = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    if now - start_time > 2.0 and not triggered and now - last_vad_check_time > 1.0 : logger.warning("No audio received from VAD input stream for ~%.1f seconds.", now - start_time); last_vad_check_time = now
                    continue
                if len(frame_bytes) != vad_bytes_per_frame: logger.warning("VAD frame unexpected size: %d bytes", len(frame_bytes)); continue
                try: is_speech = self._vad.is_speech(frame_bytes, self._vad_sample_rate)
                except Exception as vad_err: logger.error("WebRTC VAD error: %s", vad_err); continue
                total_frames += 1; last_vad_check_time = now
                if is_speech:
                    if not triggered: logger.debug("VAD triggered."); triggered = True
                    recorded_frames_bytes.append(frame_bytes); consecutive_silence_frames = 0
                elif triggered:
                    recorded_frames_bytes.append(frame_bytes); consecutive_silence_frames += 1
                    logger.log(5, "Silence frame count: %d/%d", consecutive_silence_frames, silence_frames_needed)
                    if consecutive_silence_frames >= silence_frames_needed and total_frames >= min_record_frames:
                        elapsed_ms = (now - start_time) * 1000
                        logger.info("Silence detected. Stopping recording. (Frames: %d, Elapsed: %.0f ms)", total_frames, elapsed_ms); break
        except sd.PortAudioError as e: logger.error("PortAudio error during VAD recording: %s", e, exc_info=True); return None
        except Exception as e: logger.error("Unexpected error during VAD recording: %s", e, exc_info=True); return None
        finally:
            if stream is not None:
                try:
                    if not stream.closed: stream.stop(); stream.close()
                    logger.debug("VAD audio stream stopped/closed.")
                except Exception as e: logger.error("Error closing VAD stream: %s", e)
        if not recorded_frames_bytes: logger.warning("VAD recording finished, but no frames were captured."); return None
        try:
            audio_data_int16 = np.frombuffer(b"".join(recorded_frames_bytes), dtype=np.int16)
            return audio_data_int16.astype(np.float32) / 32767.0
        except Exception as e: logger.error("Failed to convert VAD bytes to numpy: %s", e); return None

    def start_async_recording(self, sample_rate: int):
        """Starts recording audio in a background thread."""
        with self._async_recording_active:
            if self._async_recording_thread is not None and self._async_recording_thread.is_alive():
                logger.warning("Async recording is already active.")
                return False
            logger.info("Starting asynchronous PTT recording at %d Hz...", sample_rate)
            self._async_sample_rate = sample_rate
            self._async_audio_buffer = []
            self._async_recording_stop_event.clear()
            self._async_recording_thread = threading.Thread(target=self._async_record_loop, args=(sample_rate,), daemon=True)
            self._async_recording_thread.start()
            return True

    def _async_record_loop(self, sample_rate: int):
        """Background thread for continuous recording."""
        block_size = 1024
        q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=100)
        def record_callback(indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
            if status: logger.warning("Async Record Callback Status: %s", str(status))
            try:
                 if isinstance(indata, np.ndarray): q.put_nowait(indata.copy())
                 else: logger.error("Async callback non-numpy: %s", type(indata))
            except queue.Full: logger.warning("Async audio queue full.")
            except Exception as e: logger.error("Error in async record callback: %s", e)

        stream: Optional[sd.InputStream] = None
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=block_size, device=self._input_device_id, callback=record_callback)
            stream.start()
            logger.debug("Async recording stream started.")
            while not self._async_recording_stop_event.is_set():
                try:
                    chunk = q.get(timeout=0.1)
                    if chunk is not None: self._async_audio_buffer.append(chunk)
                except queue.Empty: continue
            logger.debug("Async recording loop received stop signal.")
        except sd.PortAudioError as e: logger.error("PortAudioError in async thread: %s", e)
        except Exception as e: logger.error("Unexpected error in async thread: %s", e, exc_info=True)
        finally:
             if stream:
                 try:
                    if not stream.closed: stream.stop(); stream.close()
                    logger.debug("Async recording stream stopped/closed.")
                 except Exception as e: logger.error("Error closing async stream: %s", e)
             logger.debug("Async recording loop finished.")

    def stop_async_recording(self) -> Optional[np.ndarray]:
        """Stops the background recording and returns audio."""
        stopped_thread = None; final_audio = None
        with self._async_recording_active:
            if self._async_recording_thread is None or not self._async_recording_thread.is_alive():
                logger.warning("Async recording not active."); return None
            logger.info("Stopping asynchronous PTT recording...")
            self._async_recording_stop_event.set()
            stopped_thread = self._async_recording_thread
        if stopped_thread:
            stopped_thread.join(timeout=1.0)
            if stopped_thread.is_alive(): logger.warning("Async thread did not stop.")
        with self._async_recording_active:
             if not self._async_audio_buffer: logger.warning("Async recording captured no audio."); final_audio = None
             else:
                 try:
                     logger.debug("Concatenating %d chunks.", len(self._async_audio_buffer))
                     final_audio = np.concatenate(self._async_audio_buffer, axis=0).flatten()
                     duration = len(final_audio) / self._async_sample_rate
                     logger.info("Async recording stopped. Duration: %.2f sec.", duration)
                 except Exception as e: logger.error("Error processing async buffer: %s", e); final_audio = None
             self._async_recording_thread = None; self._async_audio_buffer = []; self._async_recording_stop_event.clear()
        return final_audio

    def play_audio( self, audio_data: np.ndarray, sample_rate: int, wait_completion: bool = True ):
        """Plays audio data."""
        if audio_data is None or audio_data.size == 0: logger.warning("Attempted to play empty audio."); return
        if not isinstance(audio_data, np.ndarray): logger.error("Invalid audio_data type: %s", type(audio_data)); return
        if audio_data.dtype != np.float32:
            logger.warning("Audio not float32 (%s), converting.", audio_data.dtype);
            try: # Simplified conversion attempt
                if np.issubdtype(audio_data.dtype, np.integer): audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                else: audio_data = audio_data.astype(np.float32)
            except Exception as e: logger.error("Failed conversion: %s",e); return

        logger.info("Playing audio (%.2f seconds, %d Hz)...", len(audio_data) / sample_rate, sample_rate)
        try:
            if self._normalize_playback:
                max_abs_val = np.max(np.abs(audio_data))
                if max_abs_val == 0: logger.warning("Audio is silent."); return
                if max_abs_val > 1.0: logger.warning("Clipping detected. Normalizing."); audio_data = audio_data / max_abs_val
            sd.play(audio_data, samplerate=sample_rate, device=self._output_device_id)
            if wait_completion: sd.wait(); logger.debug("Audio playback finished.")
        except sd.PortAudioError as e: logger.error("PortAudio playback error: %s", e); raise AudioError(...) from e
        except Exception as e: logger.error("Unexpected playback error: %s", e); raise AudioError(...) from e

    def stop_playback(self):
        """Stops any currently playing audio."""
        logger.info("Stopping audio playback."); sd.stop()

    @staticmethod
    def save_wave(filepath: str, audio_data: np.ndarray, sample_rate: int):
        """Saves a NumPy audio array to a WAV file."""
        if audio_data is None or audio_data.size == 0: logger.warning("Attempted save empty audio: %s", filepath); return
        logger.debug("Saving audio to %s (%d Hz)", filepath, sample_rate)
        try:
             if audio_data.dtype == np.float32:
                 audio_data = np.clip(audio_data, -1.0, 1.0); audio_int16 = (audio_data * 32767).astype(np.int16)
             elif audio_data.dtype == np.int16: audio_int16 = audio_data
             else: logger.error("Unsupported dtype for WAV: %s", audio_data.dtype); raise AudioError(...)
             Path(filepath).parent.mkdir(parents=True, exist_ok=True)
             with wave.open(filepath, 'wb') as wf:
                 wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
                 wf.writeframes(audio_int16.tobytes())
             logger.info("Audio saved successfully: %s", filepath)
        except IOError as e: logger.error("Failed write WAV %s: %s", filepath, e); raise AudioError(...) from e
        except Exception as e: logger.error("Unexpected WAV save error %s: %s", filepath, e); raise AudioError(...) from e