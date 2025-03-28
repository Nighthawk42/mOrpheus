# modules/llm_manager.py

import time
import requests
import json
from typing import Optional, List, Dict, Any

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
            "llm.base_url": "http://127.0.0.1:1234/v1",
            "llm.request_timeout_sec": 20.0,
            "llm.max_retries": 3,
            "llm.chat.endpoint": "/chat/completions",
            "llm.chat.model": "local-model", # Placeholder
            "llm.chat.system_prompt": "You are a helpful assistant.",
            "llm.chat.max_tokens": 300,
            "llm.chat.temperature": 0.7,
            "llm.chat.top_p": 0.9,
            "llm.chat.repetition_penalty": 1.1,
        }
        nested_keys = key_path.split('.')
        val = defaults
        try:
            for k in nested_keys: val = val[k]
            return val
        except KeyError:
            return default
    # Mock PerformanceMonitor
    class MockPerformanceMonitor:
        def start_timer(self, name): pass
        def stop_timer(self, name, record_count=True): pass
        def record_event(self, name, count=1.0): pass
    # performance_monitor = MockPerformanceMonitor() # Example instantiation

# --- Constants ---
DEFAULT_ERROR_RESPONSE = "I'm having trouble connecting to my brain right now. Please try again in a moment."
TIMEOUT_RESPONSE = "I need a little more time to think about that. Could you ask again?"

class LLMError(Exception):
    """Custom exception for LLM interaction errors."""
    pass

class LLMManager:
    """Handles interactions with the Language Model API (e.g., LM Studio)."""

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        logger.info("Initializing LLMManager...")
        self._perf_monitor = performance_monitor

        # --- Load LLM Configuration ---
        self._base_url: str = get_setting("llm.base_url", "http://127.0.0.1:1234/v1").rstrip('/')
        self._timeout: float = get_setting("llm.request_timeout_sec", 20.0)
        self._max_retries: int = get_setting("llm.max_retries", 3)

        # --- Chat Specific Configuration ---
        self._chat_endpoint: str = get_setting("llm.chat.endpoint", "/chat/completions").lstrip('/')
        self._chat_model: str = get_setting("llm.chat.model", "local-model")
        self._system_prompt: str = get_setting("llm.chat.system_prompt", "You are a helpful assistant.")
        self._chat_max_tokens: int = get_setting("llm.chat.max_tokens", 300)
        self._chat_temperature: float = get_setting("llm.chat.temperature", 0.7)
        self._chat_top_p: float = get_setting("llm.chat.top_p", 0.9)
        self._chat_repetition_penalty: float = get_setting("llm.chat.repetition_penalty", 1.1)

        # --- Request Setup ---
        self._session = requests.Session()
        self._headers = {"Content-Type": "application/json"}
        self._chat_url = f"{self._base_url}/{self._chat_endpoint}"

        logger.info("LLMManager configured for URL: %s", self._chat_url)
        logger.info("Chat Model: %s", self._chat_model)


    def generate_chat_response(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Sends input to the LLM chat endpoint and returns the generated response.

        Args:
            user_input: The latest input from the user.
            chat_history: (Optional) A list of previous messages in the conversation,
                          following the OpenAI format: [{"role": "user/assistant", "content": "..."}, ...]

        Returns:
            The assistant's generated response (str), or a default error message if generation fails.
        """
        if not user_input:
            logger.warning("Received empty user input for chat.")
            return "I didn't catch that. Could you please repeat?"

        # Construct messages payload
        messages = [{"role": "system", "content": self._system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})

        # Construct request payload according to LM Studio / OpenAI API format
        payload = {
            "model": self._chat_model,
            "messages": messages,
            "max_tokens": self._chat_max_tokens,
            "temperature": self._chat_temperature,
            "top_p": self._chat_top_p,
            "repeat_penalty": self._chat_repetition_penalty, # Note: LM Studio uses 'repeat_penalty'
            "stream": False # We want the complete response here
            # Add other parameters if supported/needed (e.g., presence_penalty, frequency_penalty)
        }

        logger.info("Sending chat request to LLM...")
        logger.debug("Chat Request Payload: %s", json.dumps(payload, indent=2)) # Log full payload at debug

        if self._perf_monitor:
            self._perf_monitor.start_timer("llm_response_time")
            self._perf_monitor.record_event("llm_requests")

        response_text = DEFAULT_ERROR_RESPONSE # Default in case of failure

        for attempt in range(self._max_retries):
            try:
                response = self._session.post(
                    self._chat_url,
                    headers=self._headers,
                    json=payload,
                    timeout=self._timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

                data = response.json()
                logger.debug("LLM Raw Response: %s", json.dumps(data, indent=2))

                # Extract response content (structure may vary slightly by LLM server)
                # Assuming OpenAI compatible structure:
                if data.get("choices") and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message")
                    if message and isinstance(message, dict):
                        response_text = message.get("content", "").strip()
                        if response_text:
                            logger.info("LLM response received successfully.")
                            # Estimate token count (very rough)
                            token_est = len(response_text.split()) * 1.33
                            logger.info("LLM Response (~%d tokens): \"%s\"",
                                         int(token_est),
                                         (response_text[:70] + '...') if len(response_text) > 70 else response_text
                                         )
                            if self._perf_monitor: self._perf_monitor.record_event("llm_output_tokens", token_est)
                            break # Success, exit retry loop
                        else:
                            logger.warning("LLM response 'content' was empty.")
                    else:
                         logger.warning("LLM response missing 'message' structure in 'choices'.")
                else:
                     logger.warning("LLM response missing 'choices' structure.")

                # If we got here, extraction failed or response was empty
                response_text = DEFAULT_ERROR_RESPONSE # Reset to error message

            except requests.exceptions.Timeout:
                logger.warning("LLM request timed out (Attempt %d/%d)", attempt + 1, self._max_retries)
                if attempt == self._max_retries - 1:
                    response_text = TIMEOUT_RESPONSE
                    if self._perf_monitor: self._perf_monitor.record_event("errors")
                    break
                # Exponential backoff delay
                delay = 1 * (2 ** attempt)
                logger.info("Retrying LLM request in %d seconds...", delay)
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                logger.error("LLM request failed (Attempt %d/%d): %s", attempt + 1, self._max_retries, e, exc_info=True)
                if self._perf_monitor: self._perf_monitor.record_event("errors")
                if attempt == self._max_retries - 1:
                    response_text = DEFAULT_ERROR_RESPONSE
                    break # Max retries reached
                delay = 1 * (2 ** attempt)
                logger.info("Retrying LLM request in %d seconds...", delay)
                time.sleep(delay)

            except json.JSONDecodeError as e:
                 logger.error("Failed to decode LLM JSON response: %s", e)
                 logger.debug("LLM Raw Response Text: %s", response.text) # Log raw text if JSON fails
                 if self._perf_monitor: self._perf_monitor.record_event("errors")
                 response_text = DEFAULT_ERROR_RESPONSE
                 break # Don't retry on decode error

            except Exception as e:
                logger.critical("Unexpected error during LLM chat request: %s", e, exc_info=True)
                if self._perf_monitor: self._perf_monitor.record_event("errors")
                response_text = DEFAULT_ERROR_RESPONSE
                # Depending on the error, might want to break or retry
                break # Break on unexpected errors for now

        # End of retry loop

        if self._perf_monitor:
            self._perf_monitor.stop_timer("llm_response_time") # Always stop timer

        return response_text

    def close_session(self):
        """Closes the underlying requests session."""
        logger.debug("Closing LLMManager requests session.")
        self._session.close()