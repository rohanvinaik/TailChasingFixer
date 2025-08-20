"""Centralized logging configuration for TailChasingFixer."""

import json
import logging
import logging.handlers
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs."""
    
    # Patterns for sensitive data
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|token|secret|password|credential)["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE)
    # Only redact absolute paths outside the project
    EXTERNAL_PATH_PATTERN = re.compile(r'(?<![\w/])(/(?:usr|home|var|tmp|opt|etc|private|Users)/[^\s\])}"\',]+)')
    
    def __init__(self, project_root: Optional[Path] = None):
        super().__init__()
        self.project_root = project_root or Path.cwd()
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact sensitive information from log records."""
        # Redact from message
        if hasattr(record, 'msg'):
            record.msg = self._redact_message(str(record.msg))
            
        # Redact from args if present
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact_message(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(self._redact_message(str(arg)) for arg in record.args)
                
        return True
    
    def _redact_message(self, message: str) -> str:
        """Redact sensitive data from a message."""
        # Redact API keys and secrets
        message = self.API_KEY_PATTERN.sub(r'\1=***REDACTED***', message)
        
        # Redact external file paths (keep project paths)
        def redact_path(match):
            path = match.group(0)
            try:
                # Check if path is within project
                path_obj = Path(path)
                if path_obj.exists() and self.project_root in path_obj.parents:
                    return path  # Keep project paths
            except:
                pass
            # Redact external paths
            parts = path.split('/')
            if len(parts) > 3:
                return f"/{parts[1]}/***REDACTED***"
            return "***REDACTED_PATH***"
            
        message = self.EXTERNAL_PATH_PATTERN.sub(redact_path, message)
        return message


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        # Add context info if present
        for key in ['user', 'session_id', 'request_id', 'duration', 'operation']:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
                
        return json.dumps(log_obj, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        if not sys.stderr.isatty():
            # No colors if not a terminal
            return super().format(record)
            
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = 'tailchasing',
    level: str = 'INFO',
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True,
    json_format: bool = True,
    redact_sensitive: bool = True,
    project_root: Optional[Path] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        console: Enable console output
        file: Enable file output
        json_format: Use JSON format for file logs
        redact_sensitive: Redact sensitive data from logs
        project_root: Project root for path redaction
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Clear any existing handlers
    logger.propagate = False  # Prevent propagation to root logger to avoid duplicates
    
    # Add sensitive data filter if enabled
    if redact_sensitive:
        sensitive_filter = SensitiveDataFilter(project_root)
        logger.addFilter(sensitive_filter)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if sys.stderr.isatty():
            # Use colored formatter for terminal
            console_formatter = ConsoleFormatter(
                '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # Use simple formatter for non-terminal
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        if log_dir is None:
            log_dir = Path.cwd() / 'logs'
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with date
        date_str = datetime.now().strftime('%Y%m%d')
        if json_format:
            log_file = log_dir / f'tcf_{date_str}.jsonl'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(JSONFormatter())
        else:
            log_file = log_dir / f'tcf_{date_str}.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get or create a logger with the standard configuration.
    
    Args:
        name: Logger name (usually __name__)
        **kwargs: Additional arguments for setup_logging
        
    Returns:
        Configured logger instance
    """
    # Check if logger already exists and is configured
    logger = logging.getLogger(name)
    if logger.handlers:
        # Ensure propagate is False even for existing loggers
        logger.propagate = False
        return logger
        
    # Set up new logger
    return setup_logging(name, **kwargs)


# Convenience functions for different verbosity levels
def get_verbose_logger(name: str) -> logging.Logger:
    """Get logger configured for verbose output (DEBUG level)."""
    return get_logger(name, level='DEBUG')


def get_quiet_logger(name: str) -> logging.Logger:
    """Get logger configured for quiet output (WARNING level)."""
    return get_logger(name, level='WARNING')


def log_operation(logger: logging.Logger, operation: str, **context):
    """
    Log an operation with context.
    
    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context to log
    """
    logger.info(f"Starting operation: {operation}", extra={'operation': operation, 'extra_fields': context})


def log_model_prompt(logger: logging.Logger, prompt: str, model: str = 'unknown'):
    """
    Log model prompts at DEBUG level.
    
    Args:
        logger: Logger instance
        prompt: The prompt being sent
        model: Model name
    """
    logger.debug(f"Model prompt to {model}", extra={'extra_fields': {'model': model, 'prompt_length': len(prompt)}})


def log_retry(logger: logging.Logger, operation: str, attempt: int, max_attempts: int, error: Optional[Exception] = None):
    """
    Log retry attempts at WARNING level.
    
    Args:
        logger: Logger instance
        operation: Operation being retried
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
        error: Optional error that caused retry
    """
    error_msg = f": {str(error)}" if error else ""
    logger.warning(
        f"Retrying {operation} (attempt {attempt}/{max_attempts}){error_msg}",
        extra={'extra_fields': {'operation': operation, 'attempt': attempt, 'max_attempts': max_attempts}}
    )