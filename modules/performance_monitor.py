# modules/performance_monitor.py

import time
from collections import defaultdict
from typing import Dict, Any

# Use logger, assuming log_manager is available
try:
    from .log_manager import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log_manager. Using default logger for PerformanceMonitor.")

class PerformanceMonitor:
    """
    Tracks various performance metrics of the virtual assistant.
    """
    def __init__(self):
        self._start_time: float = time.monotonic()
        self._metrics: Dict[str, Any] = defaultdict(float)
        self._timers: Dict[str, float] = {} # For tracking durations
        logger.info("Performance monitor initialized.")

    def record_event(self, event_name: str, count: float = 1.0):
        """
        Increments a counter for a specific event.

        Examples: 'llm_requests', 'tts_requests', 'hotword_detections', 'errors'
        """
        self._metrics[event_name] += count
        logger.debug("Event recorded: %s (+%.1f)", event_name, count)

    def start_timer(self, timer_name: str):
        """
        Starts a timer for a specific operation.

        Examples: 'transcription_time', 'llm_response_time', 'tts_synthesis_time'
        """
        self._timers[timer_name] = time.monotonic()
        logger.debug("Timer started: %s", timer_name)

    def stop_timer(self, timer_name: str, record_count: bool = True):
        """
        Stops a timer and records the duration in milliseconds.

        Args:
            timer_name: The name of the timer to stop (must match start_timer).
            record_count: If True, also increments a counter named f"{timer_name}_count".
        """
        if timer_name in self._timers:
            end_time = time.monotonic()
            duration_ms = (end_time - self._timers[timer_name]) * 1000
            # Store total duration and count to calculate average later
            total_duration_key = f"{timer_name}_total_ms"
            count_key = f"{timer_name}_count"

            self._metrics[total_duration_key] += duration_ms
            if record_count:
                self._metrics[count_key] += 1

            logger.debug(
                "Timer stopped: %s, Duration: %.2f ms", timer_name, duration_ms
            )
            del self._timers[timer_name] # Remove timer once stopped
        else:
            logger.warning("Attempted to stop timer '%s' that was not started.", timer_name)

    def set_value(self, metric_name: str, value: Any):
        """Sets a specific metric to a given value (e.g., current model name)."""
        self._metrics[metric_name] = value
        logger.debug("Metric set: %s = %s", metric_name, value)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns a copy of the current metrics."""
        # Add overall uptime
        metrics_copy = self._metrics.copy()
        metrics_copy["uptime_seconds"] = time.monotonic() - self._start_time
        return metrics_copy

    def get_summary(self) -> str:
        """Generates a formatted string summary of key performance indicators."""
        metrics = self.get_metrics()
        uptime_sec = metrics.get("uptime_seconds", 0)
        llm_reqs = metrics.get("llm_requests", 0)
        tts_reqs = metrics.get("tts_requests", 0)
        stt_reqs = metrics.get("stt_requests", 0)
        errors = metrics.get("errors", 0)

        summary = f"Uptime: {time.strftime('%H:%M:%S', time.gmtime(uptime_sec))}"
        summary += f" | LLM: {int(llm_reqs)}"
        summary += f" | TTS: {int(tts_reqs)}"
        summary += f" | STT: {int(stt_reqs)}"
        summary += f" | Errors: {int(errors)}"

        # Add average times if available
        for timer_base in ["transcription_time", "llm_response_time", "tts_synthesis_time"]:
            total_ms = metrics.get(f"{timer_base}_total_ms", 0)
            count = metrics.get(f"{timer_base}_count", 0)
            if count > 0:
                avg_ms = total_ms / count
                summary += f" | Avg {timer_base.split('_')[0].upper()}: {avg_ms:.0f}ms"

        return summary

    def log_summary(self):
        """Logs the performance summary using the configured logger."""
        logger.info("Performance Summary: %s", self.get_summary())