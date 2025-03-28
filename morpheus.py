#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mOrpheus - A Voice Assistant Framework
"""

import argparse
import sys
import time
import threading
from typing import Optional, List, Dict, Any

# --- Rich Imports ---
import numpy as np
from rich.console import Console
from rich.status import Status
from rich.panel import Panel
from rich.text import Text

# --- Core Manager Imports ---
try:
    from modules.config_manager import load_config, get_setting, ConfigError
    from modules.log_manager import setup_logging, logger # Use the logger configured by log_manager
    from modules.performance_monitor import PerformanceMonitor
    from modules.audio_manager import AudioManager, AudioError
    from modules.stt_manager import STTManager, STTError, EXPECTED_SAMPLE_RATE as STT_SAMPLE_RATE
    from modules.llm_manager import LLMManager, LLMError
    from modules.tts_manager import TTSManager, TTSError, SNAC_EXPECTED_RATE as TTS_SAMPLE_RATE
    from modules.hotword_manager import HotwordManager, HotwordError
except ImportError as e:
    print(f"FATAL: Failed to import core modules: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Console Object ---
# Use this for printing user interaction (You/Assistant) separately from logs
console = Console()

class VirtualAssistant:
    """
    Orchestrates the voice assistant's components and main interaction loop.
    """
    def __init__(self, config_path: Optional[str] = None):
        """Initializes all managers and loads configuration."""
        # --- Config & Logging First ---
        # Wrap initial setup for better error reporting if logging fails
        try:
            self._config_data = load_config(config_path)
            log_level = get_setting("general.log_level", "INFO")
            # Setup logging (will use RichHandler for console via log_manager)
            setup_logging(log_level=log_level)
        except ConfigError as e:
            print(f"FATAL: Configuration Error: {e}", file=sys.stderr); raise
        except Exception as e:
            print(f"FATAL: Initial config/logging error: {e}", file=sys.stderr); raise ConfigError(f"Failed setup: {e}") from e

        # --- Log Startup Info ---
        logger.info("-" * 50)
        logger.info("Initializing mOrpheus Virtual Assistant...")
        logger.info("Configuration loaded.")

        # --- Initialize Managers ---
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.audio_manager: Optional[AudioManager] = None
        # ... (rest are similar)
        self.stt_manager: Optional[STTManager] = None; self.llm_manager: Optional[LLMManager] = None
        self.tts_manager: Optional[TTSManager] = None; self.hotword_manager: Optional[HotwordManager] = None
        try:
            self.performance_monitor = PerformanceMonitor()
            self.audio_manager = AudioManager()
            self.stt_manager = STTManager(performance_monitor=self.performance_monitor)
            self.llm_manager = LLMManager(performance_monitor=self.performance_monitor)
            self.tts_manager = TTSManager(performance_monitor=self.performance_monitor)
            self.hotword_manager = HotwordManager(performance_monitor=self.performance_monitor)
            # Store interaction settings
            self.interaction_mode: str = get_setting("general.interaction_mode", "push_to_talk")
            self.post_response_delay: float = get_setting("general.post_response_delay_sec", 0.5)
            self.is_vad_enabled: bool = get_setting("audio.vad.enabled", False) # Store VAD status

            # --- Log Key Settings Using Rich Panel ---
            stt_model = get_setting("stt.model_size", "N/A")
            llm_model = get_setting("llm.chat.model", "N/A")
            tts_model = get_setting("tts.model", "N/A")
            hotword_status = "Enabled" if self.hotword_manager.is_enabled else "Disabled"
            if self.hotword_manager.is_enabled:
                hotwords = get_setting("hotword.models", [])
                hotword_status += f" ({', '.join(hotwords)})"

            settings_summary = Text.assemble(
                ("STT Model: ", "bold cyan"), (stt_model, "white"), "\n",
                ("LLM Model: ", "bold cyan"), (llm_model, "white"), "\n",
                ("TTS Model: ", "bold cyan"), (tts_model, "white"), "\n",
                ("VAD: ", "bold cyan"), ("Enabled" if self.is_vad_enabled else "Disabled", "white"), "\n",
                ("Hotword: ", "bold cyan"), (hotword_status, "white"), "\n",
                ("Interaction: ", "bold cyan"), (self.interaction_mode, "white")
            )
            console.print(Panel(settings_summary, title="[bold]Configuration[/bold]", border_style="dim blue", expand=False))

            logger.info("All managers initialized successfully.")

        except (AudioError, STTError, TTSError, HotwordError) as e:
            logger.critical("Failed manager init: %s", e, exc_info=True); raise
        except Exception as e:
            logger.critical("Unexpected init error: %s", e, exc_info=True); raise RuntimeError(f"Manager init failed: {e}") from e

        # --- State Variables ---
        self._running = threading.Event(); self._running.clear()
        self._stop_called = False

    def run(self):
        """Starts the main interaction loop of the assistant."""
        if self._running.is_set(): logger.warning("Assistant already running."); return
        if not all([self.audio_manager, ...]): logger.critical(...); return # Null checks

        self._running.set(); self._stop_called = False
        console.print(Panel(f"🎤 Assistant Activated | Mode: [cyan]{self.interaction_mode}[/cyan] | VAD: {'[green]On[/green]' if self.is_vad_enabled else '[yellow]Off[/yellow]'} ",
                           title="[bold green]mOrpheus[/bold green]", border_style="green", expand=False))

        # Start hotword manager if applicable (VAD must be enabled for hotword/both modes)
        if self.is_vad_enabled and (self.interaction_mode == "hotword" or self.interaction_mode == "both"):
            if self.hotword_manager and self.hotword_manager.is_enabled:
                 logger.info("Starting hotword listener...")
                 self.hotword_manager.start()
            else: logger.warning("Hotword interaction mode selected, but hotword is disabled/failed.")
        elif (self.interaction_mode == "hotword" or self.interaction_mode == "both") and not self.is_vad_enabled:
            logger.warning("Hotword/Both mode requires VAD to be enabled. Interaction may not work as expected.")

        # --- Main Loop with Rich Status ---
        try:
            # Create a status object outside the loop to update it
            with console.status("", spinner="dots") as status:
                while self._running.is_set():
                    audio_data: Optional[np.ndarray] = None
                    try:
                        # 1. Wait for Activation
                        status.update("[bold cyan]Waiting for activation...[/bold cyan] (Press ENTER or say hotword)")
                        activated, activation_method = self._wait_for_activation()
                        if not activated or not self._running.is_set(): break

                        console.print(f"▶️ Activated via [yellow]{activation_method}[/yellow]!")

                        # --- Interaction Cycle ---
                        # 2. Record Audio
                        if self.is_vad_enabled:
                            status.update("🎙️ Listening... (VAD active)", spinner="simpleDotsScrolling")
                            audio_data = self.audio_manager.record_audio(target_sample_rate=STT_SAMPLE_RATE)
                        elif self.interaction_mode == "push_to_talk":
                            status.update("🔴 Recording... (Press [bold]ENTER[/bold] to stop)", spinner="recording")
                            if not self.audio_manager.start_async_recording(STT_SAMPLE_RATE):
                                logger.error("Failed to start async PTT recording.")
                                continue
                            stop_key_pressed = self._wait_for_stop_keypress(status) # Pass status to update
                            audio_data = self.audio_manager.stop_async_recording()
                            if not stop_key_pressed:
                                logger.warning("Recording stop not via keypress.")
                                if not self._running.is_set(): break
                                audio_data = None # Discard partial
                        else:
                            logger.error("Cannot record: VAD off & mode != push_to_talk.")
                            continue

                        if audio_data is None or audio_data.size == 0:
                            logger.warning("No audio captured."); continue

                        # Ensure managers exist (redundant check, but safe)
                        if not all([self.stt_manager, self.llm_manager, self.tts_manager, self.audio_manager]):
                            logger.critical("Manager missing mid-cycle!"); self._running.clear(); break

                        # 3. Transcribe
                        status.update("📝 Transcribing...", spinner="bouncingBar")
                        user_text, _, _ = self.stt_manager.transcribe_audio(audio_data, STT_SAMPLE_RATE)
                        if not user_text: logger.warning("Transcription empty."); continue
                        console.print(Text.assemble("👤 You: ", (user_text, "bright_blue"))) # Use Rich Text

                        # 4. LLM
                        status.update("🧠 Thinking...", spinner="line")
                        response_text = self.llm_manager.generate_chat_response(user_text)
                        if not response_text: logger.error("LLM failed."); continue
                        console.print(Text.assemble("🤖 Asst: ", (response_text, "green"))) # Use Rich Text

                        # 5. TTS
                        status.update("🔊 Synthesizing response...", spinner="material")
                        _filepath, audio_response, _rate = self.tts_manager.synthesize_speech(response_text)
                        if audio_response is None: logger.error("TTS failed."); continue

                        # 6. Play
                        status.update("💬 Speaking...", spinner="dots")
                        self.audio_manager.play_audio(audio_response, TTS_SAMPLE_RATE, wait_completion=True)

                        # 7. Delay
                        if self.post_response_delay > 0:
                             status.update(f"Cooldown ({self.post_response_delay}s)...", spinner="clock")
                             time.sleep(self.post_response_delay)

                        # Status will automatically reset to "Waiting..." at the start of the next loop

                    # --- Error Handling within Loop ---
                    # Log errors, but allow loop to continue to wait for next activation
                    except AudioError as e: logger.error("Audio Error in loop: %s", e); time.sleep(1)
                    except STTError as e: logger.error("STT Error in loop: %s", e); time.sleep(1)
                    except LLMError as e: logger.error("LLM Error in loop: %s", e); time.sleep(1)
                    except TTSError as e: logger.error("TTS Error in loop: %s", e); time.sleep(1)
                    except Exception as e:
                        logger.critical("Unexpected critical error in main loop: %s", e, exc_info=True)
                        if self.performance_monitor: self.performance_monitor.record_event("critical_errors")
                        self._running.clear(); break # Stop loop on critical errors

        finally:
            logger.info("Main loop terminated.")
            self.stop() # Ensure cleanup happens

    def _wait_for_activation(self) -> tuple[bool, Optional[str]]:
        """
        Waits for hotword (if VAD enabled) or first PTT keypress.
        Does NOT display prompts, only checks for activation signals.
        """
        check_interval = 0.1
        # VAD status is checked once in run() now

        while self._running.is_set():
            # --- Check Hotword ---
            if self.is_vad_enabled and (self.interaction_mode == "hotword" or self.interaction_mode == "both") and \
               self.hotword_manager and self.hotword_manager.is_enabled and self.hotword_manager.is_running:
                detected_keyword = self.hotword_manager.get_detected_keyword()
                if detected_keyword: return True, f"Hotword ({detected_keyword})"

            # --- Check PTT Keypress ---
            if self.interaction_mode == "push_to_talk" or self.interaction_mode == "both":
                 if self._check_for_keypress():
                      self._flush_stdin()
                      method = "Push-to-talk Start" if not self.is_vad_enabled and self.interaction_mode == "push_to_talk" else "Push-to-talk"
                      return True, method

            try: time.sleep(check_interval)
            except KeyboardInterrupt: logger.info("Interrupt during activation wait."); self._running.clear(); return False, None
        return False, None

    def _wait_for_stop_keypress(self, status: Status) -> bool:
         """
         Waits for Enter press to stop async recording. Updates status.
         Returns True if Enter was pressed, False otherwise (interrupt/error).
         """
         # No need for separate debug log, status handles the prompt
         # We don't use input() here to avoid interfering with status display.
         # Rely on the non-blocking check instead.
         check_interval = 0.05 # Check more frequently
         while self._running.is_set():
             if self._check_for_keypress():
                  logger.debug("Stop keypress detected.")
                  self._flush_stdin()
                  status.update("⏹️ Recording stopped.", spinner="dots") # Briefly update status
                  time.sleep(0.1) # Short pause to show status
                  return True # Stop key pressed successfully

             try: time.sleep(check_interval)
             except KeyboardInterrupt: logger.info("Interrupt while waiting for stop keypress."); self._running.clear(); return False
             except Exception as e: logger.error("Error waiting for stop keypress: %s", e); self._running.clear(); return False
         return False # Exited loop


    def _check_for_keypress(self) -> bool:
        """Non-blocking check for keypress."""
        # ... (Implementation remains the same) ...
        try: import msvcrt; return msvcrt.kbhit()
        except ImportError:
            try: import select, sys; return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
            except: return False # Catch potential errors like closed stdin
        except Exception: return False


    def _flush_stdin(self):
        """Flush any lingering characters from standard input."""
        # ... (Implementation remains the same) ...
        try: import termios, sys; termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except ImportError:
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except: pass # Ignore errors
        except Exception: pass


    def stop(self):
        """Signals the assistant to stop and cleans up resources."""
        # ... (Implementation remains the same, uses self._stop_called flag) ...
        if self._stop_called: return
        if not self._running.is_set(): self._stop_called = True; return # If not running, just set flag

        logger.info("Initiating mOrpheus shutdown...")
        self._running.clear()
        self._stop_called = True

        logger.debug("Stopping Hotword Manager...")
        if self.hotword_manager and self.hotword_manager.is_running: self.hotword_manager.stop()

        logger.debug("Stopping Async Audio Recording (if active)...")
        try:
             if self.audio_manager:
                  if getattr(self.audio_manager, '_async_recording_thread', None) is not None:
                       self.audio_manager.stop_async_recording()
        except Exception as e_stop_async: logger.warning("Ignoring error during async audio stop: %s", e_stop_async)

        logger.debug("Stopping Audio Playback...")
        if self.audio_manager: self.audio_manager.stop_playback()

        logger.debug("Closing Network Sessions...")
        if self.llm_manager: self.llm_manager.close_session()
        if self.tts_manager: self.tts_manager.close_session()

        if self.performance_monitor:
            logger.info("-" * 50); self.performance_monitor.log_summary(); logger.info("-" * 50)

        # Use console.print for final styled message
        console.print(Panel("[bold red]mOrpheus Assistant Deactivated[/bold red]", border_style="red", expand=False))
        logger.info("=" * 50)


# --- Main Execution ---
def main():
    """Parses arguments, initializes, and runs the VirtualAssistant."""
    # ... (Argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description="Start the mOrpheus Virtual Assistant.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML.")
    args = parser.parse_args()

    assistant: Optional[VirtualAssistant] = None
    exit_code = 0
    try:
        assistant = VirtualAssistant(config_path=args.config)
        assistant.run() # Blocks until finished/interrupted

    except ConfigError as e: # Catch config/init errors
        # Logger might not be fully available, rely on print for critical startup failures
        print(f"\nFATAL CONFIGURATION ERROR: {e}\n", file=sys.stderr)
        exit_code = 1
    except (AudioError, STTError, TTSError, HotwordError, RuntimeError) as e:
         # Catch manager init errors
         print(f"\nFATAL INITIALIZATION ERROR: {e}\n", file=sys.stderr)
         # Try logging if available
         try: logger.critical("Initialization Error: %s", e, exc_info=True)
         except NameError: pass
         exit_code = 1
    except KeyboardInterrupt:
        console.print("\n[yellow]User interrupt detected. Exiting.[/yellow]")
        # Assistant.run() likely already called stop(), but call again if object exists
        # if assistant and not getattr(assistant, '_stop_called', False): assistant.stop() # Redundant if finally works
        exit_code = 0 # Normal exit for Ctrl+C
    except Exception as e:
        console.print(f"\n[bold red]UNEXPECTED FATAL ERROR:[/bold red]")
        # Print traceback using rich console
        console.print_exception(show_locals=False) # Set show_locals=True for more debug info
        exit_code = 1
    finally:
        # Ensure stop is attempted if assistant was created, unless already called
        if assistant and not getattr(assistant, '_stop_called', False):
            logger.debug("Ensuring assistant stop called in main finally block.")
            assistant.stop()

    # Use console for final message if logger might be broken
    if exit_code == 0:
         console.print("[green]mOrpheus shutdown complete.[/green]")
    else:
         console.print("[red]mOrpheus shutdown with errors.[/red]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()