# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Kiro Gateway — FastAPI application factory and server-side components.

This module contains the ASGI application instance, lifespan management,
middleware configuration, and route registration. It is the core of the
server, separated from CLI logic to support installation as a uv tool.

Usage:
    # Import the app object for ASGI servers
    from kiro.app import app

    # Import for validation before startup
    from kiro.app import validate_configuration
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from kiro.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    REFRESH_TOKEN,
    KIRO_CREDS_FILE,
    KIRO_CLI_DB_FILE,
    LOG_LEVEL,
    STREAMING_READ_TIMEOUT,
    VPN_PROXY_URL,
    USER_CONFIG_FILE,
    ACCOUNT_SYSTEM,
    ACCOUNTS_CONFIG_FILE,
    ACCOUNTS_STATE_FILE,
)
from kiro.account_manager import AccountManager
from kiro.routes_openai import router as openai_router
from kiro.routes_anthropic import router as anthropic_router
from kiro.exceptions import validation_exception_handler
from kiro.debug_middleware import DebugLoggerMiddleware


# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """Intercepts logs from standard logging and redirects them to loguru.

    This allows capturing logs from uvicorn, FastAPI and other libraries
    that use standard logging instead of loguru.

    Also filters out noisy shutdown-related exceptions (CancelledError,
    KeyboardInterrupt) that are normal during Ctrl+C but uvicorn logs
    as ERROR.
    """

    # Exceptions that are normal during shutdown and should not be logged as errors
    SHUTDOWN_EXCEPTIONS = (
        "CancelledError",
        "KeyboardInterrupt",
        "asyncio.exceptions.CancelledError",
    )

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by forwarding it to loguru.

        Args:
            record: The log record from standard logging.
        """
        # Filter out shutdown-related exceptions that uvicorn logs as ERROR
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type is not None:
                exc_name = exc_type.__name__
                if exc_name in self.SHUTDOWN_EXCEPTIONS:
                    logger.info("Server shutdown in progress...")
                    return

        # Also filter by message content for cases where exc_info is not set
        msg = record.getMessage()
        if any(exc in msg for exc in self.SHUTDOWN_EXCEPTIONS):
            return

        # Get the corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller frame for correct source display
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging_intercept() -> None:
    """Configure log interception from standard logging to loguru.

    Intercepts logs from:
    - uvicorn (access logs, error logs)
    - uvicorn.error
    - uvicorn.access
    - fastapi
    """
    loggers_to_intercept = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ]

    for logger_name in loggers_to_intercept:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


# Configure uvicorn/fastapi log interception
setup_logging_intercept()


# ==================================================================================================
# VPN/Proxy Configuration
# ==================================================================================================
# Must be set BEFORE creating any httpx clients (including in lifespan)
# httpx automatically picks up HTTP_PROXY, HTTPS_PROXY, ALL_PROXY from environment

if VPN_PROXY_URL:
    # Normalize URL - add http:// if no scheme specified
    _proxy_url_with_scheme = VPN_PROXY_URL if "://" in VPN_PROXY_URL else f"http://{VPN_PROXY_URL}"

    # Set environment variables for httpx to pick up automatically
    os.environ['HTTP_PROXY'] = _proxy_url_with_scheme
    os.environ['HTTPS_PROXY'] = _proxy_url_with_scheme
    os.environ['ALL_PROXY'] = _proxy_url_with_scheme

    # Exclude localhost from proxy to avoid routing local requests through it
    _no_proxy_hosts = os.environ.get("NO_PROXY", "")
    _local_hosts = "127.0.0.1,localhost"
    if _no_proxy_hosts:
        os.environ["NO_PROXY"] = f"{_no_proxy_hosts},{_local_hosts}"
    else:
        os.environ["NO_PROXY"] = _local_hosts

    logger.info(f"Proxy configured: {_proxy_url_with_scheme}")
    logger.debug(f"NO_PROXY: {os.environ['NO_PROXY']}")


def _print_config_errors(errors: list[str]) -> None:
    """Print configuration errors in a formatted block.

    Args:
        errors: List of error message strings to display.
    """
    logger.error("")
    logger.error("=" * 60)
    logger.error("  CONFIGURATION ERROR")
    logger.error("=" * 60)
    for error in errors:
        for line in error.split('\n'):
            logger.error(f"  {line}")
    logger.error("=" * 60)
    logger.error("")


def validate_configuration(silent: bool = False) -> bool:
    """Validate that required configuration is present.

    Checks that at least one credential source is configured:
    REFRESH_TOKEN, KIRO_CREDS_FILE, or KIRO_CLI_DB_FILE.

    Args:
        silent: If True, suppress error output. Useful when the caller
                will handle the failure itself (e.g. launch a setup wizard).

    Returns:
        True if configuration is valid, False otherwise.
        Callers are responsible for deciding what to do on False
        (e.g. launch the setup wizard or exit).
    """
    # Account System takes priority: credentials.json existing means the
    # legacy .env variables are not required.
    if Path(ACCOUNTS_CONFIG_FILE).exists():
        logger.debug(f"Found {ACCOUNTS_CONFIG_FILE}, skipping legacy .env validation")
        return True

    errors = []

    has_refresh_token = bool(os.environ.get("REFRESH_TOKEN", REFRESH_TOKEN))
    _creds_file = os.environ.get("KIRO_CREDS_FILE", KIRO_CREDS_FILE)
    _cli_db = os.environ.get("KIRO_CLI_DB_FILE", KIRO_CLI_DB_FILE)

    has_creds_file = bool(_creds_file)
    has_cli_db = bool(_cli_db)

    if _creds_file:
        creds_path = Path(_creds_file).expanduser()
        if not creds_path.exists():
            has_creds_file = False
            logger.warning(f"KIRO_CREDS_FILE not found: {_creds_file}")

    if _cli_db:
        cli_db_path = Path(_cli_db).expanduser()
        if not cli_db_path.exists():
            has_cli_db = False
            logger.warning(f"KIRO_CLI_DB_FILE not found: {_cli_db}")

    if not has_refresh_token and not has_creds_file and not has_cli_db:
        errors.append(
            "No Kiro credentials configured!\n"
            "\n"
            f"Run 'kiro-gateway config --edit' to set up your credentials.\n"
            "\n"
            "Or set environment variables directly:\n"
            "   REFRESH_TOKEN=\"your_refresh_token\"\n"
            "   KIRO_CREDS_FILE=\"/path/to/credentials.json\"\n"
            "   KIRO_CLI_DB_FILE=\"~/.local/share/kiro-cli/data.sqlite3\"\n"
            "\n"
            f"Config file location: {USER_CONFIG_FILE}"
        )

    if errors:
        if not silent:
            _print_config_errors(errors)
        return False

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifecycle.

    Creates and initializes:
    - Shared HTTP client with connection pooling
    - AccountManager (credentials.json migration, per-account auth and models)
    - The first working account, so at least one is ready before requests

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to the application after startup.
    """
    logger.info("Starting application... Creating state managers.")

    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0
    )
    timeout = httpx.Timeout(
        connect=30.0,
        read=STREAMING_READ_TIMEOUT,
        write=30.0,
        pool=30.0
    )
    app.state.http_client = httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        follow_redirects=True
    )
    logger.info("Shared HTTP client created with connection pooling")

    
    # ==============================================================================
    # Legacy Fallback: .env → credentials.json
    # ==============================================================================
    creds_path = Path(ACCOUNTS_CONFIG_FILE)
    
    # Check if we have legacy .env credentials
    has_refresh_token = bool(REFRESH_TOKEN)
    has_creds_file = bool(KIRO_CREDS_FILE) and Path(KIRO_CREDS_FILE).expanduser().exists()
    has_cli_db = bool(KIRO_CLI_DB_FILE) and Path(KIRO_CLI_DB_FILE).expanduser().exists()
    
    # Helper function to add optional per-account overrides from .env
    def _add_env_overrides(entry: dict) -> None:
        """Add optional per-account overrides from .env (only if set)"""
        profile_arn = os.getenv("PROFILE_ARN")
        if profile_arn:
            entry["profile_arn"] = profile_arn
        
        region = os.getenv("KIRO_REGION")
        if region:
            entry["region"] = region
        
        api_region = os.getenv("KIRO_API_REGION")
        if api_region:
            entry["api_region"] = api_region
    
    if ACCOUNT_SYSTEM:
        # Account system enabled: create credentials.json ONCE (migration)
        if not creds_path.exists():
            if has_refresh_token or has_creds_file or has_cli_db:
                logger.info("credentials.json not found, creating from .env (one-time migration)")
                credentials = []
                
                # Priority: SQLite DB > JSON file > environment variables (same as KiroAuthManager)
                if has_cli_db:
                    entry = {
                        "type": "sqlite",
                        "path": KIRO_CLI_DB_FILE
                    }
                    _add_env_overrides(entry)
                    credentials.append(entry)
                elif has_creds_file:
                    entry = {
                        "type": "json",
                        "path": KIRO_CREDS_FILE
                    }
                    _add_env_overrides(entry)
                    credentials.append(entry)
                elif has_refresh_token:
                    entry = {
                        "type": "refresh_token",
                        "refresh_token": REFRESH_TOKEN
                    }
                    _add_env_overrides(entry)
                    credentials.append(entry)
            
                # Save credentials.json
                with open(creds_path, 'w', encoding='utf-8') as f:
                    json.dump(credentials, f, indent=2, ensure_ascii=False)
                
                logger.info("Created credentials.json from .env (one-time migration)")
    else:
        # Legacy mode: ALWAYS recreate credentials.json from .env
        if has_refresh_token or has_creds_file or has_cli_db:
            logger.debug("Legacy mode: recreating credentials.json from .env")
            credentials = []
            
            # Priority: SQLite DB > JSON file > environment variables (same as KiroAuthManager)
            if has_cli_db:
                entry = {
                    "type": "sqlite",
                    "path": KIRO_CLI_DB_FILE
                }
                _add_env_overrides(entry)
                credentials.append(entry)
            elif has_creds_file:
                entry = {
                    "type": "json",
                    "path": KIRO_CREDS_FILE
                }
                _add_env_overrides(entry)
                credentials.append(entry)
            elif has_refresh_token:
                entry = {
                    "type": "refresh_token",
                    "refresh_token": REFRESH_TOKEN
                }
                _add_env_overrides(entry)
                credentials.append(entry)
            
            # Save credentials.json (overwrite if exists)
            with open(creds_path, 'w', encoding='utf-8') as f:
                json.dump(credentials, f, indent=2, ensure_ascii=False)
            
            logger.debug("credentials.json recreated from .env (legacy mode)")
    
    # ==============================================================================
    # Create AccountManager
    # ==============================================================================
    app.state.account_manager = AccountManager(
        credentials_file=ACCOUNTS_CONFIG_FILE,
        state_file=ACCOUNTS_STATE_FILE
    )
    
    # Load credentials and state
    await app.state.account_manager.load_credentials()
    await app.state.account_manager.load_state()
    
    # Store account_system flag
    app.state.account_system = ACCOUNT_SYSTEM
    
    # ==============================================================================
    # Initialize first working account (blocking)
    # ==============================================================================
    all_accounts = list(app.state.account_manager._accounts.keys())
    
    if not all_accounts:
        logger.error("No accounts configured in credentials.json")
        raise RuntimeError("No accounts configured in credentials.json")
    
    # Determine start index from state.json
    start_index = app.state.account_manager._current_account_index
    
    # Try to initialize accounts (full circle)
    initialized = False
    
    for i in range(len(all_accounts)):
        current_index = (start_index + i) % len(all_accounts)
        account_id = all_accounts[current_index]
        
        logger.info(f"Attempting to initialize account: {account_id}")
        
        success = await app.state.account_manager._initialize_account(account_id)
        
        if success:
            logger.info(f"Successfully initialized account: {account_id}")
            initialized = True
            break
        else:
            logger.warning(f"Failed to initialize account: {account_id}")
    
    if not initialized:
        logger.error("Failed to initialize any account. Check your credentials.")
        raise RuntimeError("Failed to initialize any account")
    
    # Save initial state
    await app.state.account_manager._save_state()
    
    # Start background task for periodic state saving
    save_task = asyncio.create_task(
        app.state.account_manager.save_state_periodically()
    )
    
    logger.info("Account system initialized successfully")
    
    yield
    
    # Graceful shutdown
    logger.info("Shutting down application...")
    
    # Cancel background task
    save_task.cancel()
    try:
        await save_task
    except asyncio.CancelledError:
        pass
    
    # Final state save
    await app.state.account_manager._save_state()
    logger.info("Final state saved")
    
    # Close HTTP client
    try:
        await app.state.http_client.aclose()
        logger.info("Shared HTTP client closed")
    except Exception as e:
        logger.warning(f"Error closing shared HTTP client: {e}")


# --- FastAPI Application ---
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Debug Logger Middleware ---
app.add_middleware(DebugLoggerMiddleware)

# --- Validation Error Handler Registration ---
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# --- Route Registration ---
app.include_router(openai_router)
app.include_router(anthropic_router)
