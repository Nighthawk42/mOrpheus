# modules/token_decoder.py

import time
from typing import Iterable, Optional, List, Generator, Dict, Tuple, Any
import logging

import numpy as np
import torch
# Ensure snac is installed, handle potential ImportError
try:
    from snac import SNAC
except ImportError:
    SNAC = None

# Use logger
try:
    from .log_manager import logger
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log_manager. Using default logger for token_decoder.")


# --- SNAC Model Loading (Done once at import time) ---
_snac_model: Optional[Any] = None
_snac_device: Optional[str] = None
_cuda_stream: Optional[torch.cuda.Stream] = None
_is_snac_initialized: bool = False

def initialize_snac():
    """Initializes the SNAC model and determines the device."""
    global _snac_model, _snac_device, _cuda_stream, _is_snac_initialized
    if _is_snac_initialized:
        return

    if SNAC is None:
        logger.error("SNAC library is not installed. TTS Token decoding will not work.")
        _is_snac_initialized = True # Mark as initialized (but failed)
        return

    try:
        logger.info("Initializing SNAC model for TTS token decoding...")
        # Check device availability
        if torch.cuda.is_available():
            _snac_device = "cuda"
            _cuda_stream = torch.cuda.Stream()
        else:
            _snac_device = "cpu"
            _cuda_stream = None
        logger.info("Using SNAC on device: %s", _snac_device)

        # Load the pre-trained SNAC model
        snac_model_name = "hubertsiuzdak/snac_24khz"
        # Suppress the specific FutureWarning from torch.load within SNAC if desired
        import warnings
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)
        _snac_model = SNAC.from_pretrained(snac_model_name).eval()
        warnings.resetwarnings() # Optional: Reset warnings filters after loading
        _snac_model = _snac_model.to(_snac_device)
        logger.info("SNAC model '%s' loaded successfully.", snac_model_name)
        _is_snac_initialized = True

    except Exception as e:
        logger.error("Failed to initialize SNAC model: %s", e, exc_info=True)
        _snac_model = None # Ensure model is None on failure
        _is_snac_initialized = True # Mark as initialized (but failed)

# --- Run initialization ---
initialize_snac()

# --- Constants (Matching Original Logic) ---
TOKENS_PER_AUDIO_FRAME = 7
SNAC_EXPECTED_RATE = 24000
AUDIO_SLICE_START = 2048
AUDIO_SLICE_END = 4096
MAX_TOKEN_ID = 4096      # For validation bounds, NOT used in modulo calculation here
MIN_FRAMES_REQUIRED = 4  # Minimum number of frames (7 tokens each) needed before processing
PROCESS_CHUNK_FRAMES = 1 # How many new frames trigger processing (usually 1)
PROCESS_WINDOW_FRAMES = MIN_FRAMES_REQUIRED # How many frames (x7 tokens) to pass to decode func
TOKEN_CACHE: Dict[Tuple[str, int], Optional[int]] = {}
MAX_CACHE_SIZE = 10000
TOKEN_ID_OFFSET = 10     # Offset from the original formula

# --- Core Decoding Functions ---

def _original_convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """
    Identical logic to the original project's convert_to_audio.
    Accepts a list of token IDs (expects PROCESS_WINDOW_FRAMES * 7 = 28).
    """
    if _snac_model is None or _snac_device is None:
        logger.error("SNAC model not initialized."); return None

    required_tokens = PROCESS_WINDOW_FRAMES * TOKENS_PER_AUDIO_FRAME
    if len(multiframe) < required_tokens:
         logger.warning("_original_convert_to_audio needs %d tokens, got %d.", required_tokens, len(multiframe))
         return None

    # Process the required window size
    frame_tokens = multiframe[:required_tokens]
    num_frames = PROCESS_WINDOW_FRAMES # Should be 4

    logger.debug("Original logic: Decoding batch of %d frames (%d tokens)...", num_frames, len(frame_tokens))

    try:
        # Prepare tensors for the batch of frames
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=_snac_device)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=_snac_device)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=_snac_device)
        # Use torch.tensor directly on the list slice for efficiency
        frame_tensor = torch.tensor(frame_tokens, dtype=torch.int32, device=_snac_device)

        # Populate code tensors using loops (as in original)
        for j in range(num_frames):
            idx = j * TOKENS_PER_AUDIO_FRAME
            codes_0[j] = frame_tensor[idx]
            codes_1[j * 2] = frame_tensor[idx + 1]
            codes_1[j * 2 + 1] = frame_tensor[idx + 4]
            codes_2[j * 4] = frame_tensor[idx + 2]
            codes_2[j * 4 + 1] = frame_tensor[idx + 3]
            codes_2[j * 4 + 2] = frame_tensor[idx + 5]
            codes_2[j * 4 + 3] = frame_tensor[idx + 6]

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)] # Add batch dim

    except (ValueError, TypeError, IndexError) as e:
         logger.error("Tensor creation failed (Original logic): %s", e)
         return None

    # --- RE-ENABLE VALIDATION - LOGGING ONLY ---
    # Check if calculated IDs (which might be negative) fall outside a "reasonable" range
    # if SNAC is expected to handle them internally. Let's check against 0 and MAX_TOKEN_ID (4096).
    validation_passed = True
    min_val_0, max_val_0 = torch.min(codes[0]).item(), torch.max(codes[0]).item()
    min_val_1, max_val_1 = torch.min(codes[1]).item(), torch.max(codes[1]).item()
    min_val_2, max_val_2 = torch.min(codes[2]).item(), torch.max(codes[2]).item()

    if min_val_0 < 0 or max_val_0 > MAX_TOKEN_ID:
        logger.warning("Validation FAIL (Log Only): codes_0 out of range [0, %d]. Min: %d, Max: %d", MAX_TOKEN_ID, min_val_0, max_val_0)
        validation_passed = False
    if min_val_1 < 0 or max_val_1 > MAX_TOKEN_ID:
        logger.warning("Validation FAIL (Log Only): codes_1 out of range [0, %d]. Min: %d, Max: %d", MAX_TOKEN_ID, min_val_1, max_val_1)
        validation_passed = False
    if min_val_2 < 0 or max_val_2 > MAX_TOKEN_ID:
        logger.warning("Validation FAIL (Log Only): codes_2 out of range [0, %d]. Min: %d, Max: %d", MAX_TOKEN_ID, min_val_2, max_val_2)
        validation_passed = False

    if not validation_passed:
         logger.debug("Problematic batch tokens for validation fail: %s", frame_tokens)
         logger.warning("Proceeding with potentially invalid tokens to debug CUDA assert...")
    # else:
    #     logger.debug("Token ID validation passed (check based on [0, %d]).", MAX_TOKEN_ID)

    # Perform decoding
    audio_bytes = None
    try:
        stream_ctx = torch.cuda.stream(_cuda_stream) if _cuda_stream is not None else torch.no_grad()
        with stream_ctx, torch.inference_mode():
            start_decode_time = time.monotonic()
            # --- Call SNAC decode ---
            audio_hat = _snac_model.decode(codes)
            decode_duration_ms = (time.monotonic() - start_decode_time) * 1000
            logger.debug("SNAC batch decode (Original logic) took %.2f ms", decode_duration_ms)

            # --- USE ORIGINAL SLICE ---
            audio_slice = audio_hat[:, :, AUDIO_SLICE_START:AUDIO_SLICE_END]
            if logger.isEnabledFor(logging.DEBUG): # Avoid potentially expensive tensor ops if not logging
                logger.debug("Original logic slice shape: %s, min=%.4f, max=%.4f",
                             audio_slice.shape, torch.min(audio_slice).item(), torch.max(audio_slice).item())

            if audio_slice.numel() == 0:
                logger.warning("audio_slice is empty after decoding!")
                return None

            # --- Conversion to bytes ---
            if _snac_device == "cuda":
                audio_int16_tensor = (audio_slice * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
                audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
            else: # CPU
                audio_np = audio_slice.detach().numpy()
                audio_int16 = (audio_np * 32767.0).clip(-32768.0, 32767.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

        logger.debug("Batch successfully converted to %d bytes.", len(audio_bytes) if audio_bytes else 0)
        return audio_bytes

    except Exception as e:
        # Catch CUDA errors here specifically if possible
        logger.error("Error during SNAC decoding step (Original logic): %s", e, exc_info=True)
        return None

# --- Reverted to ORIGINAL formula ---
def _turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Parses custom token string using the original formula."""
    token_string = token_string.strip()
    if "<custom_token_" not in token_string: return None
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1 or not token_string.endswith(">"): return None
    try:
        number_str = token_string[last_token_start + 14:-1]
        # Original formula:
        token_id = int(number_str) - TOKEN_ID_OFFSET - ((index % TOKENS_PER_AUDIO_FRAME) * MAX_TOKEN_ID)
        # logger.debug("Original Formula: Raw %s -> ID %d (Index %d)", number_str, token_id, index)
        return token_id
    except (ValueError, IndexError):
        logger.warning("Failed parse token ID (Original): '%s'", token_string); return None


# --- Generator using IDENTICAL logic to original tokens_decoder ---
def decode_tts_tokens(token_gen: Iterable[str]) -> Generator[bytes, None, None]:
    """Identical logic to the original project's tokens_decoder generator."""
    if not _is_snac_initialized or _snac_model is None:
        logger.error("SNAC model not available."); return

    buffer: List[int] = []
    count = 0
    min_tokens_required = MIN_FRAMES_REQUIRED * TOKENS_PER_AUDIO_FRAME # 28
    process_every_tokens = PROCESS_CHUNK_FRAMES * TOKENS_PER_AUDIO_FRAME # 7
    process_window_tokens = PROCESS_WINDOW_FRAMES * TOKENS_PER_AUDIO_FRAME # 28

    processed_stream_tokens = 0
    yielded_chunks = 0
    logger.debug("Starting TTS token decoding stream (Strict Original Logic)...")

    for token_text in token_gen:
        processed_stream_tokens += 1
        if processed_stream_tokens % 100 == 0:
             logger.debug("Processing token #%d from stream: %s...", processed_stream_tokens, token_text[:30])

        # --- Get Token ID (Original Formula) ---
        # Use 'count' for index as in original logic
        cache_key = (token_text, count % TOKENS_PER_AUDIO_FRAME)
        if cache_key in TOKEN_CACHE:
            token_id = TOKEN_CACHE[cache_key]
        else:
            token_id = _turn_token_into_id(token_text, count) # Pass current count as index
            if len(TOKEN_CACHE) < MAX_CACHE_SIZE:
                TOKEN_CACHE[cache_key] = token_id # Cache result (including None)

        # --- USE ORIGINAL > 0 CHECK ---
        if token_id is not None and token_id > 0:
            buffer.append(token_id)
            count += 1 # Increment count *only* for valid tokens > 0 added

            # --- Original Condition to Process ---
            if count % process_every_tokens == 0 and count >= min_tokens_required:
                buffer_to_proc = buffer[-process_window_tokens:] # Last 28 tokens
                logger.debug("Processing original window (count=%d, buffer_len=%d, window_len=%d)",
                             count, len(buffer), len(buffer_to_proc))

                # Call the function with original logic structure
                audio_samples = _original_convert_to_audio(buffer_to_proc, count)

                if audio_samples is not None:
                    yield audio_samples
                    yielded_chunks += 1
                    if yielded_chunks % 5 == 0: # Log every 5 batches yielded
                         logger.debug("Yielded audio batch #%d (Original Logic)", yielded_chunks)
        # else:
             # Optionally log if token_id was None or <= 0
             # logger.debug("Skipping token: ID=%s", token_id)


    logger.debug("Finished consuming token stream (%d total tokens processed).", processed_stream_tokens)
    logger.debug("Finished TTS token decoding stream (%d total audio batches yielded).", yielded_chunks)


# decode_tts_tokens_to_bytes remains the same
def decode_tts_tokens_to_bytes(token_stream: Iterable[str]) -> bytes:
    """Decodes token stream and concatenates audio bytes."""
    if not _is_snac_initialized or _snac_model is None: return b""
    # Use the generator version which now incorporates the original logic
    audio_segments = list(decode_tts_tokens(token_stream))
    if not audio_segments: logger.warning("Token decoding yielded no audio segments."); return b""
    return b"".join(audio_segments)