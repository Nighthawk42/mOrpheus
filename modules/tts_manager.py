# modules/tts_manager.py

import os
import re
import time
import json
import wave
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any, Tuple
import logging # Correctly imported

import requests
import numpy as np

# Use logger, config manager, performance monitor, and token decoder
try:
    from .log_manager import logger
    from .config_manager import get_setting
    from .performance_monitor import PerformanceMonitor
    from .token_decoder import decode_tts_tokens_to_bytes, SNAC_EXPECTED_RATE, _is_snac_initialized as is_snac_ready
    from .audio_manager import AudioManager
except ImportError:
    # import logging # Already imported above
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log/config/perf/token/audio managers. Using defaults.")
    # Mock dependencies... (Assuming mocks are correctly defined as before)
    def get_setting(key_path: str, default: Optional[Any] = None) -> Any:
        # ... (mock implementation) ...
        pass
    class MockPerformanceMonitor: # ... (mock implementation) ...
        pass
    def decode_tts_tokens_to_bytes(stream): return b""
    SNAC_EXPECTED_RATE = 24000
    is_snac_ready = True
    class MockAudioManager: # ... (mock implementation) ...
        @staticmethod
        def save_wave(filepath, audio_data, sample_rate): pass
    AudioManager = MockAudioManager


# --- Constants ---
TTS_OUTPUT_FILENAME_FORMAT = "{voice}_{timestamp}.wav"
COMBINED_FILENAME_FORMAT = "{voice}_{timestamp}_combined.wav"

class TTSError(Exception):
    """Custom exception for TTS related errors."""
    pass


def _clean_text_for_tts(text: str) -> str:
    """Cleans text before sending to the TTS engine."""
    # ... (Implementation remains the same) ...
    text = str(text); text = text.replace('\n', ' '); text = re.sub(r'\*+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text); text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _segment_text(text: str, max_words: int) -> List[str]:
    """Splits text into segments, trying to respect sentence boundaries."""
    # ... (Implementation remains the same) ...
    if not text: return []
    if max_words <= 0: return [text]
    words = text.split();
    if len(words) <= max_words: return [text]
    segments = []; current_segment_words: List[str] = []
    sentence_ending_punctuation = (".", "!", "?", ";", ":", ".\"","!\"","?\"")
    min_segment_len_factor = 0.1; merge_overshoot_factor = 1.2
    for word in words:
        current_segment_words.append(word)
        word_ends_sentence = any(word.endswith(p) for p in sentence_ending_punctuation)
        current_length = len(current_segment_words)
        if current_length >= max_words or \
           (word_ends_sentence and current_length > max_words * 0.6):
            segments.append(" ".join(current_segment_words)); current_segment_words = []
    if current_segment_words: segments.append(" ".join(current_segment_words))
    merged_segments: List[str] = []; i = 0
    while i < len(segments):
         current = segments[i]; current_len = len(current.split())
         if i == len(segments) - 1 or \
            len(segments[i+1].split()) > max_words * min_segment_len_factor or \
            current_len > max_words * min_segment_len_factor:
             merged_segments.append(current); i += 1
         else:
              next_segment = segments[i+1]; merged = current + " " + next_segment; merged_len = len(merged.split())
              if merged_len <= max_words * merge_overshoot_factor:
                  merged_segments.append(merged); logger.debug("Merged short segment."); i += 2
              else: merged_segments.append(current); i += 1
    if not merged_segments and segments: return segments
    logger.debug("Segmented text into %d parts.", len(merged_segments))
    return merged_segments


class TTSManager:
    """Handles Text-to-Speech synthesis using an LLM endpoint and SNAC decoding."""

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        # ... (Initialization remains the same) ...
        logger.info("Initializing TTSManager...")
        self._perf_monitor = performance_monitor
        if not is_snac_ready: raise TTSError("SNAC model failed init or not found.")
        self._base_url: str = get_setting("llm.base_url", "...").rstrip('/')
        self._timeout: float = get_setting("llm.request_timeout_sec", 120.0)
        self._max_retries: int = get_setting("llm.max_retries", 2)
        self._tts_endpoint: str = get_setting("tts.endpoint", "/completions").lstrip('/')
        self._tts_model: str = get_setting("tts.model", "orpheus-model")
        self._default_voice: str = get_setting("tts.default_voice", "tara")
        self._tts_max_tokens: int = get_setting("tts.max_tokens", 4096)
        self._tts_temperature: float = get_setting("tts.temperature", 0.6)
        self._tts_top_p: float = get_setting("tts.top_p", 0.9)
        self._tts_repetition_penalty: float = get_setting("tts.repetition_penalty", 1.0)
        self._tts_speed: float = get_setting("tts.speed", 1.0)
        self._output_dir = Path(get_setting("tts.output_dir", "outputs"))
        self._clear_output: bool = get_setting("tts.clear_output_on_start", True)
        self._segment_max_words: int = get_setting("tts.segmentation.max_words_per_segment", 60)
        self._session = requests.Session()
        self._headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        self._tts_url = f"{self._base_url}/{self._tts_endpoint}"
        self._prepare_output_directory()
        logger.info("TTSManager configured for URL: %s", self._tts_url)
        logger.info("TTS Model: %s | Default Voice: %s", self._tts_model, self._default_voice)

    def _prepare_output_directory(self):
        """Creates the output directory and optionally clears it."""
        # ... (Implementation remains the same) ...
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("TTS output directory: %s", self._output_dir.resolve())
            if self._clear_output:
                logger.info("Clearing previous TTS output files from %s...", self._output_dir)
                count = 0; deleted_files = []
                for item in self._output_dir.glob('*.wav'):
                    try: item.unlink(); deleted_files.append(item.name); count += 1
                    except OSError as e: logger.warning("Could not delete %s: %s", item, e)
                if count > 0: logger.debug("Cleared files: %s", ", ".join(deleted_files))
                logger.info("Cleared %d previous WAV files.", count)
        except Exception as e: logger.error("Failed prepare output dir '%s': %s", self._output_dir, e); raise TTSError(f"Output dir error: {e}") from e

    # --- synthesize_speech with Segmentation ---
    def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[np.ndarray], int]:
        """Synthesizes speech, handling segmentation for long text."""
        if not text: logger.warning("Synthesize speech called with empty text."); return None, None, SNAC_EXPECTED_RATE

        active_voice = voice if voice else self._default_voice
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', active_voice):
             logger.warning("Invalid voice tag '%s'. Using default '%s'.", active_voice, self._default_voice); active_voice = self._default_voice

        cleaned_text = _clean_text_for_tts(text)
        if not cleaned_text: logger.warning("Text empty after cleaning: '%s'", text); return None, None, SNAC_EXPECTED_RATE

        segments = _segment_text(cleaned_text, self._segment_max_words)
        if len(segments) > 1: logger.info("Text is long, processing %d segments...", len(segments))
        else: logger.info("Processing single text segment...")

        segment_audio_data: List[np.ndarray] = []
        segment_files: List[Path] = []
        total_success = True

        # --- Process Segments ---
        for i, segment_text in enumerate(segments):
            logger.info("Synthesizing segment %d/%d...", i + 1, len(segments))
            if logger.isEnabledFor(logging.DEBUG): # Correct check using imported logging
                 seg_preview = (segment_text[:60] + '...') if len(segment_text) > 60 else segment_text
                 logger.debug("Segment text: \"%s\"", seg_preview)

            if self._perf_monitor: self._perf_monitor.start_timer("tts_synthesis_time"); self._perf_monitor.record_event("tts_requests")
            audio_bytes: Optional[bytes] = None
            try:
                audio_bytes = self._synthesize_segment(segment_text, active_voice)
                if audio_bytes:
                    if len(audio_bytes) % 2 != 0: logger.warning("Odd bytes (%d) seg %d. Trimming.", len(audio_bytes), i+1); audio_bytes = audio_bytes[:-1]
                    if not audio_bytes: logger.warning("Audio empty post-trim seg %d.", i+1); continue
                    audio_segment_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    segment_audio_data.append(audio_segment_np)
                    duration_sec = len(audio_segment_np) / SNAC_EXPECTED_RATE
                    logger.info("Segment %d synthesized successfully (%.2f seconds).", i + 1, duration_sec)
                    if len(segments) > 1:
                         timestamp = int(time.time_ns() // 1_000_000)
                         seg_filename = self._output_dir / f"{active_voice}_{timestamp}_seg{i}.wav"
                         try: AudioManager.save_wave(str(seg_filename), audio_segment_np, SNAC_EXPECTED_RATE)
                         except Exception as save_e: logger.error("Failed save segment %s: %s", seg_filename, save_e)
                         segment_files.append(seg_filename)
                else: logger.error("Failed synthesize segment %d.", i + 1); total_success = False; break
            except TTSError as e: logger.error("TTS Error seg %d: %s", i + 1, e); total_success = False; break
            except Exception as e: logger.error("Unexpected Error seg %d: %s", i + 1, e, exc_info=True); total_success = False; break
            finally:
                 if self._perf_monitor:
                     try: self._perf_monitor.stop_timer("tts_synthesis_time")
                     # CORRECTED SYNTAX: except Exception as t_e:
                     except Exception as t_e: logger.error("Timer error: %s", t_e)
            if len(segments) > 1 and i < len(segments) - 1: time.sleep(0.2)

        # --- Combine & Save ---
        if not total_success or not segment_audio_data:
            logger.error("TTS synthesis failed. No audio generated.");
            for f in segment_files:
                try:
                    f.unlink()
                except OSError:
                    pass
            return None, None, SNAC_EXPECTED_RATE
        try: final_audio_data = np.concatenate(segment_audio_data)
        except ValueError as e: logger.error("Failed concat segments: %s", e); return None, None, SNAC_EXPECTED_RATE
        if final_audio_data.size == 0: logger.error("Concatenated audio data is empty."); return None, None, SNAC_EXPECTED_RATE

        timestamp = int(time.time())
        is_multi_segment = len(segments) > 1
        output_filename = (COMBINED_FILENAME_FORMAT if is_multi_segment else TTS_OUTPUT_FILENAME_FORMAT).format(voice=active_voice, timestamp=timestamp)
        output_filepath = self._output_dir / output_filename
        try: AudioManager.save_wave(str(output_filepath), final_audio_data, SNAC_EXPECTED_RATE)
        # CORRECTED SYNTAX: except Exception as e:
        except Exception as e: logger.error("Failed save final TTS %s: %s", output_filepath, e)
        if is_multi_segment:
             logger.debug("Cleaning up %d intermediate segment files...", len(segment_files))
             for f in segment_files:
                 try: f.unlink()
                 # CORRECTED SYNTAX: except OSError as e:
                 except OSError as e: logger.warning("Could not delete temp %s: %s", f, e)
        return str(output_filepath), final_audio_data, SNAC_EXPECTED_RATE

    # --- _synthesize_segment ---
    def _synthesize_segment(self, text_segment: str, voice: str) -> Optional[bytes]:
        """Sends a single text segment to the TTS API and returns decoded audio bytes."""
        prompt = f"<|audio|>{voice}: {text_segment}<|eot_id|>"
        payload = {
            "model": self._tts_model, "prompt": prompt, "max_tokens": self._tts_max_tokens,
            "temperature": self._tts_temperature, "top_p": self._tts_top_p,
            "repeat_penalty": self._tts_repetition_penalty, "speed": self._tts_speed,
            "stream": True
        }
        log_payload = payload.copy()
        log_payload["prompt"] = (prompt[:50] + "...") if len(prompt) > 50 else prompt
        logger.debug("TTS Request Payload: %s", json.dumps(log_payload))

        audio_bytes: Optional[bytes] = None
        response: Optional[requests.Response] = None

        for attempt in range(self._max_retries + 1):
             should_retry = False
             try:
                response = self._session.post(self._tts_url, headers=self._headers, json=payload, stream=True, timeout=self._timeout)
                response.raise_for_status()
                def token_generator() -> Generator[str, None, None]:
                    nonlocal response
                    resp_to_close = response
                    try:
                         if resp_to_close is None: logger.error("BUG: token_generator started with None response"); return
                         logger.debug("Reading token stream from response...")
                         lines_processed = 0
                         for line in resp_to_close.iter_lines():
                              lines_processed += 1
                              if not line: continue
                              decoded_line = line.decode("utf-8")
                              if decoded_line.startswith("data: "):
                                   data_str = decoded_line[len("data: "):].strip()
                                   if data_str == "[DONE]": logger.debug("SSE [DONE] received."); break
                                   try: data = json.loads(data_str); token_text = data.get("choices", [{}])[0].get("text", "")
                                   except (json.JSONDecodeError, IndexError, KeyError): logger.warning("Failed decode/parse SSE JSON: %s", data_str); continue
                                   if token_text: yield token_text
                         logger.debug("Finished reading token stream (%d lines processed).", lines_processed)
                    except requests.exceptions.ChunkedEncodingError as chunk_err: logger.warning("Stream connection broken during read: %s", chunk_err)
                    # CORRECTED SYNTAX: except Exception as gen_err:
                    except Exception as gen_err: logger.error("Error reading token stream: %s", gen_err, exc_info=True)
                    finally:
                         if resp_to_close:
                              try: resp_to_close.close(); logger.debug("Closed response stream in generator finally.")
                              # CORRECTED SYNTAX: except Exception as close_e:
                              except Exception as close_e: logger.warning("Error closing response in generator: %s", close_e)
                         # response = None # Keep outer response to allow outer finally to close if needed
                logger.debug("Starting token decoding for segment...")
                start_decode_io = time.monotonic()
                audio_bytes = decode_tts_tokens_to_bytes(token_generator())
                decode_io_duration = time.monotonic() - start_decode_io
                logger.debug("Token decoding finished (%.2f sec). Got %d audio bytes.", decode_io_duration, len(audio_bytes) if audio_bytes else 0)
                if audio_bytes: break
                else: logger.warning("Decoder returned empty audio (Attempt %d/%d).", attempt + 1, self._max_retries + 1); should_retry = attempt < self._max_retries
             except requests.exceptions.Timeout: logger.warning("TTS request timed out (Attempt %d/%d)", attempt + 1, self._max_retries + 1); should_retry = attempt < self._max_retries
             except requests.exceptions.RequestException as e: logger.warning("TTS request failed (Attempt %d/%d): %s", attempt + 1, self._max_retries + 1, e); should_retry = attempt < self._max_retries
             except Exception as e: logger.error("Unexpected error during TTS segment synthesis: %s", e, exc_info=True); break
             finally:
                  if response:
                      try: response.close(); logger.debug("Closed response in outer finally.")
                      # CORRECTED SYNTAX: except Exception: pass
                      except Exception: pass
                      response = None # Mark as closed

             if should_retry: time.sleep(1 * (2 ** attempt)); logger.info("Retrying TTS segment synthesis...")
             else: break
        if not audio_bytes and ('e' not in locals() or isinstance(e, (requests.exceptions.Timeout, requests.exceptions.RequestException))):
             raise TTSError(f"TTS segment synthesis failed after {self._max_retries + 1} attempts (no audio bytes decoded).")
        return audio_bytes

    def close_session(self):
        """Closes the underlying requests session."""
        logger.debug("Closing TTSManager session."); self._session.close()