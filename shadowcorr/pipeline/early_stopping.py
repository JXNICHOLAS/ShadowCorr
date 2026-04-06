"""
Graceful early-stopping via keyboard (backtick key).

Press ` (backtick) during training to set a global stop flag that the
training loop checks at each epoch boundary.
"""

import select
import sys as system_module
import threading

_STOP_TRAINING = False
_STOP_LOCK = threading.Lock()
_KEYBOARD_THREAD = None
_KEYBOARD_THREAD_RUNNING = False


def _keyboard_listener():
    """Background thread: monitors stdin for the backtick key."""
    global _STOP_TRAINING, _KEYBOARD_THREAD_RUNNING
    try:
        if not system_module.stdin.isatty():
            try:
                import termios
                import tty
                old_settings = termios.tcgetattr(system_module.stdin)
                tty.setraw(system_module.stdin.fileno())
                while _KEYBOARD_THREAD_RUNNING:
                    try:
                        if select.select([system_module.stdin], [], [], 0.1)[0]:
                            key = system_module.stdin.read(1)
                            if key == '`':
                                with _STOP_LOCK:
                                    _STOP_TRAINING = True
                                print("\nStop signal received (backtick pressed).")
                                break
                    except (IOError, OSError):
                        break
                termios.tcsetattr(system_module.stdin, termios.TCSADRAIN, old_settings)
            except (ImportError, AttributeError):
                while _KEYBOARD_THREAD_RUNNING:
                    try:
                        if select.select([system_module.stdin], [], [], 0.1)[0]:
                            key = system_module.stdin.read(1)
                            if key == '`':
                                with _STOP_LOCK:
                                    _STOP_TRAINING = True
                                print("\nStop signal received (backtick pressed).")
                                break
                    except (IOError, OSError):
                        break
        else:
            while _KEYBOARD_THREAD_RUNNING:
                try:
                    if select.select([system_module.stdin], [], [], 0.1)[0]:
                        key = system_module.stdin.read(1)
                        if key == '`':
                            with _STOP_LOCK:
                                _STOP_TRAINING = True
                            print("\nStop signal received (backtick pressed).")
                            break
                except (IOError, OSError):
                    break
    except Exception:
        pass


def start_keyboard_monitoring():
    """Start background thread to monitor keyboard input."""
    global _KEYBOARD_THREAD, _KEYBOARD_THREAD_RUNNING
    if _KEYBOARD_THREAD is None or not _KEYBOARD_THREAD.is_alive():
        _KEYBOARD_THREAD_RUNNING = True
        _KEYBOARD_THREAD = threading.Thread(target=_keyboard_listener, daemon=True)
        _KEYBOARD_THREAD.start()


def stop_keyboard_monitoring():
    """Stop background thread monitoring keyboard input."""
    global _KEYBOARD_THREAD_RUNNING, _KEYBOARD_THREAD
    _KEYBOARD_THREAD_RUNNING = False
    if _KEYBOARD_THREAD is not None:
        _KEYBOARD_THREAD.join(timeout=0.5)
        _KEYBOARD_THREAD = None


def should_stop() -> bool:
    """Return True if the user has requested early stopping."""
    global _STOP_TRAINING
    with _STOP_LOCK:
        return _STOP_TRAINING


def reset_stop_flag():
    """Reset the stop flag before starting a new training run."""
    global _STOP_TRAINING
    with _STOP_LOCK:
        _STOP_TRAINING = False
