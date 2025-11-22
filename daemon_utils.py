"""
Daemon utilities for running the application as a background service.
"""
import os
import sys
import signal
import atexit
import time
import json
from pathlib import Path


# Global shutdown flag
shutdown_requested = False
reload_requested = False


def get_default_pid_file():
    """
    Get the default PID file path based on whether running as root.
    
    Returns:
        str: Path to PID file
    """
    if os.geteuid() == 0:
        # Running as root, use system directory
        return '/var/run/cv-caddy.pid'
    else:
        # Running as regular user, use current directory
        return os.path.join(os.getcwd(), 'cv-caddy.pid')


def write_pid_file(pid_file_path):
    """
    Write the current process ID to a file.
    
    Args:
        pid_file_path: Path to PID file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        pid_dir = os.path.dirname(pid_file_path)
        if pid_dir and not os.path.exists(pid_dir):
            os.makedirs(pid_dir, exist_ok=True)
        
        with open(pid_file_path, 'w') as f:
            f.write(str(os.getpid()))
        
        # Register cleanup function
        atexit.register(remove_pid_file, pid_file_path)
        return True
    except Exception as e:
        print(f"Error writing PID file: {e}", file=sys.stderr)
        return False


def read_pid_file(pid_file_path):
    """
    Read the process ID from a file.
    
    Args:
        pid_file_path: Path to PID file
    
    Returns:
        int: Process ID, or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(pid_file_path):
            return None
        
        with open(pid_file_path, 'r') as f:
            pid_str = f.read().strip()
            return int(pid_str)
    except (ValueError, IOError):
        return None


def remove_pid_file(pid_file_path):
    """
    Remove the PID file.
    
    Args:
        pid_file_path: Path to PID file
    """
    try:
        if os.path.exists(pid_file_path):
            os.remove(pid_file_path)
    except Exception:
        pass  # Ignore errors when removing PID file


def is_process_running(pid):
    """
    Check if a process with the given PID is running.
    
    Args:
        pid: Process ID
    
    Returns:
        bool: True if process is running, False otherwise
    """
    try:
        # Send signal 0 to check if process exists
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def send_signal_to_daemon(pid_file_path, signal_type):
    """
    Send a signal to the daemon process.
    
    Args:
        pid_file_path: Path to PID file
        signal_type: Signal to send (signal.SIGTERM, signal.SIGHUP, etc.)
    
    Returns:
        bool: True if signal sent successfully, False otherwise
    """
    pid = read_pid_file(pid_file_path)
    if pid is None:
        print(f"Error: PID file not found at {pid_file_path}")
        return False
    
    if not is_process_running(pid):
        print(f"Error: Process {pid} is not running (stale PID file)")
        remove_pid_file(pid_file_path)
        return False
    
    try:
        os.kill(pid, signal_type)
        return True
    except OSError as e:
        print(f"Error sending signal to process {pid}: {e}")
        return False


def setup_signal_handlers(shutdown_callback=None, reload_callback=None):
    """
    Set up signal handlers for graceful shutdown and reload.
    
    Args:
        shutdown_callback: Function to call on shutdown (SIGTERM, SIGINT)
        reload_callback: Function to call on reload (SIGHUP)
    """
    global shutdown_requested, reload_requested
    
    def shutdown_handler(signum, frame):
        """Handle shutdown signals."""
        global shutdown_requested
        shutdown_requested = True
        if shutdown_callback:
            shutdown_callback()
    
    def reload_handler(signum, frame):
        """Handle reload signal."""
        global reload_requested
        reload_requested = True
        if reload_callback:
            reload_callback()
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGHUP, reload_handler)


def daemonize(pid_file_path=None, stdout_log=None, stderr_log=None):
    """
    Daemonize the current process using double-fork method.
    
    Args:
        pid_file_path: Path to PID file (default: get_default_pid_file())
        stdout_log: Path to stdout log file (default: cv-caddy-daemon.log in current directory)
        stderr_log: Path to stderr log file (default: cv-caddy-daemon-error.log in current directory)
    
    Returns:
        bool: True if daemonization successful, False otherwise
    """
    if pid_file_path is None:
        pid_file_path = get_default_pid_file()
    
    # Check if already running
    existing_pid = read_pid_file(pid_file_path)
    if existing_pid and is_process_running(existing_pid):
        print(f"Error: Daemon is already running (PID: {existing_pid})")
        return False
    
    # Store current working directory and convert log paths to absolute before chdir
    original_cwd = os.getcwd()
    
    # Convert log file paths to absolute if they're relative
    if stdout_log is None:
        stdout_log = os.path.join(original_cwd, 'cv-caddy-daemon.log')
    elif not os.path.isabs(stdout_log):
        stdout_log = os.path.join(original_cwd, stdout_log)
    
    if stderr_log is None:
        stderr_log = os.path.join(original_cwd, 'cv-caddy-daemon-error.log')
    elif not os.path.isabs(stderr_log):
        stderr_log = os.path.join(original_cwd, stderr_log)
    
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            sys.exit(0)
    except OSError as e:
        print(f"Error: First fork failed: {e}", file=sys.stderr)
        return False
    
    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)
    
    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            sys.exit(0)
    except OSError as e:
        print(f"Error: Second fork failed: {e}", file=sys.stderr)
        return False
    
    # Redirect stdout and stderr
    try:
        si = open('/dev/null', 'r')
        so = open(stdout_log, 'a+')
        se = open(stderr_log, 'a+')
        
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        
        si.close()
        so.close()
        se.close()
    except Exception as e:
        print(f"Error redirecting file descriptors: {e}", file=sys.stderr)
        return False
    
    # Write PID file
    if not write_pid_file(pid_file_path):
        return False
    
    return True


def check_shutdown_requested():
    """
    Check if shutdown has been requested.
    
    Returns:
        bool: True if shutdown requested, False otherwise
    """
    return shutdown_requested


def check_reload_requested():
    """
    Check if reload has been requested.
    
    Returns:
        bool: True if reload requested, False otherwise
    """
    return reload_requested


def clear_reload_request():
    """Clear the reload request flag."""
    global reload_requested
    reload_requested = False


def get_default_status_file():
    """
    Get the default status file path based on whether running as root.
    
    Returns:
        str: Path to status file
    """
    if os.geteuid() == 0:
        # Running as root, use system directory
        return '/var/run/cv-caddy.status'
    else:
        # Running as regular user, use current directory
        return os.path.join(os.getcwd(), 'cv-caddy.status')


def update_status_file(status_file_path, status_data):
    """
    Update the status file with current daemon status.
    
    Args:
        status_file_path: Path to status file
        status_data: Dictionary with status information
    """
    try:
        status_data['timestamp'] = time.time()
        status_data['pid'] = os.getpid()
        
        status_dir = os.path.dirname(status_file_path)
        if status_dir and not os.path.exists(status_dir):
            os.makedirs(status_dir, exist_ok=True)
        
        with open(status_file_path, 'w') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        # Don't fail if status file can't be written
        pass


def read_status_file(status_file_path):
    """
    Read the status file.
    
    Args:
        status_file_path: Path to status file
    
    Returns:
        dict: Status data, or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(status_file_path):
            return None
        
        with open(status_file_path, 'r') as f:
            return json.load(f)
    except (ValueError, IOError, json.JSONDecodeError):
        return None


def systemd_notify(message=None, status=None, errno=None, bus_error=None, pid=None, uid=None, gid=None, fdname=None, fds=None, main_pid=None, ready=None, stopping=None, reloading=None, watchdog=None, watchdog_usec=None, extend_timeout_usec=None):
    """
    Send notification to systemd using sd_notify protocol.
    
    Args:
        message: Status message
        status: Status string
        ready: Set to True to notify systemd that service is ready
        stopping: Set to True to notify systemd that service is stopping
        reloading: Set to True to notify systemd that service is reloading
        watchdog: Set to True to update watchdog timestamp
        main_pid: Main PID of the service
        fds: File descriptor names
        fdname: File descriptor name
        errno: Error number
        bus_error: D-Bus error
        pid: Process ID
        uid: User ID
        gid: Group ID
        extend_timeout_usec: Extend timeout in microseconds
    
    Returns:
        bool: True if notification sent successfully, False otherwise
    """
    notify_socket = os.environ.get('NOTIFY_SOCKET')
    if not notify_socket:
        return False
    
    # Remove leading @ if present (abstract socket)
    if notify_socket.startswith('@'):
        notify_socket = '\0' + notify_socket[1:]
    
    try:
        import socket as sock
        sock_obj = sock.socket(sock.AF_UNIX, sock.SOCK_DGRAM)
        
        # Build notification message
        parts = []
        if ready is not None:
            parts.append(f"READY={1 if ready else 0}")
        if status:
            parts.append(f"STATUS={status}")
        if message:
            parts.append(f"STATUS={message}")
        if stopping is not None:
            parts.append(f"STOPPING={1 if stopping else 0}")
        if reloading is not None:
            parts.append(f"RELOADING={1 if reloading else 0}")
        if watchdog is not None:
            parts.append(f"WATCHDOG={1 if watchdog else 0}")
        if main_pid is not None:
            parts.append(f"MAINPID={main_pid}")
        if errno is not None:
            parts.append(f"ERRNO={errno}")
        if bus_error:
            parts.append(f"BUSERROR={bus_error}")
        if pid is not None:
            parts.append(f"PID={pid}")
        if uid is not None:
            parts.append(f"UID={uid}")
        if gid is not None:
            parts.append(f"GID={gid}")
        if fdname:
            parts.append(f"FDNAME={fdname}")
        if fds:
            parts.append(f"FDS={fds}")
        if extend_timeout_usec is not None:
            parts.append(f"EXTEND_TIMEOUT_USEC={extend_timeout_usec}")
        
        if not parts:
            return False
        
        notification = '\n'.join(parts)
        sock_obj.sendto(notification.encode('utf-8'), notify_socket)
        sock_obj.close()
        return True
    except Exception:
        return False

