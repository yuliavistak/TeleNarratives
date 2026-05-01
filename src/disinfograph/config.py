"""Configuration management for DisinfoGraph.

This module handles all configuration, including sensitive credentials.
All sensitive data should be stored in .env file (which is gitignored).
Never commit .env files or hardcode credentials in source code.
"""
import os
import csv
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Load environment variables from .env file
# This must be done at module import time, before any config functions are called
try:
    from dotenv import load_dotenv
    
    # Load .env from project root (where this package is installed)
    # Try multiple locations to be robust
    env_paths = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent.parent / ".env",  # Project root
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            break
    else:
        # If no .env found, try loading from current directory anyway
        # (dotenv will silently fail if file doesn't exist)
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        
except ImportError:
    # python-dotenv is optional but recommended
    import warnings
    warnings.warn(
        "python-dotenv not installed. Install it with: pip install python-dotenv\n"
        "Environment variables must be set manually or via system environment.",
        UserWarning
    )


def get_channels_csv_path(base_dir: Optional[Path] = None) -> Path:
    """Get the path to the channels CSV file."""
    if base_dir is None:
        base_dir = Path.cwd()
    return base_dir / "config" / "channels.csv"


def load_channels_from_csv(csv_path: Optional[Path] = None) -> Tuple[List[str], Dict[str, str]]:
    """Load channels and labels from CSV file.
    
    CSV format expected:
    - Column 1: Channel name (optional, for reference)
    - Column 2: Nickname (channel username with or without @)
    # - Column 3: Label (e.g., 'contain', 'disinfo', 'normal')
    
    Args:
        csv_path: Path to CSV file. If None, uses default location.
        
    Returns:
        Tuple of (list of channel usernames, dict mapping username to label)
    """
    if csv_path is None:
        csv_path = get_channels_csv_path()
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Channels CSV file not found: {csv_path}\n"
            f"Please create a CSV file with columns: Channel, Nickname, Label"
        )
    
    channels = []
    channel_labels = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get nickname (remove @ if present)
            nickname = row.get('Nickname', '').strip()
            if nickname.startswith('@'):
                nickname = nickname[1:]
            
            if not nickname:
                continue
            
            # Get label
            label = row.get('Label', '').strip().lower()
            
            channels.append(nickname)
            channel_labels[nickname.lower()] = label
    
    return channels, channel_labels


def get_channel_labels(csv_path: Optional[Path] = None) -> Dict[str, str]:
    """Get channel labels mapping from CSV file.
    
    Args:
        csv_path: Path to CSV file. If None, uses default location.
        
    Returns:
        Dictionary mapping channel username (lowercase) to label
    """
    _, labels = load_channels_from_csv(csv_path)
    return labels


def get_channel_list(csv_path: Optional[Path] = None) -> List[str]:
    """Get list of channel usernames from CSV file.
    
    Args:
        csv_path: Path to CSV file. If None, uses default location.
        
    Returns:
        List of channel usernames (without @)
    """
    channels, _ = load_channels_from_csv(csv_path)
    return channels


def get_telegram_config() -> Dict:
    """Get Telegram API configuration from environment variables.
    
    Reads from:
    - Environment variables (TELEGRAM_API_ID, TELEGRAM_API_HASH)
    - .env file in project root (recommended)
    
    Raises:
        ValueError: If required credentials are not set
        
    Returns:
        Dictionary with 'api_id' (int) and 'api_hash' (str)
    """
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    
    if not api_id or not api_hash:
        raise ValueError(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables must be set.\n"
            "Create a .env file in the project root with:\n"
            "  TELEGRAM_API_ID=your_api_id\n"
            "  TELEGRAM_API_HASH=your_api_hash\n"
            "Get your credentials from: https://my.telegram.org"
        )
    
    try:
        api_id_int = int(api_id)
    except ValueError:
        raise ValueError(f"TELEGRAM_API_ID must be a valid integer, got: {api_id}")
    
    return {
        "api_id": api_id_int,
        "api_hash": api_hash,
    }


def get_neo4j_config() -> Dict:
    """Get Neo4j configuration from environment variables.
    
    Reads from:
    - Environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
    - .env file in project root (recommended)
    - Defaults to local Neo4j if not set (bolt://localhost:7687)
    
    Raises:
        ValueError: If password is not set (required even for local)
        
    Returns:
        Dictionary with 'uri', 'username', 'password', and 'database'
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Default to local Neo4j
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    if not password:
        raise ValueError(
            "NEO4J_PASSWORD environment variable must be set.\n"
            "For local Neo4j, create a .env file in the project root with:\n"
            "  NEO4J_URI=bolt://localhost:7687  (or leave unset for default)\n"
            "  NEO4J_USERNAME=neo4j  (or leave unset for default)\n"
            "  NEO4J_PASSWORD=your_local_password\n"
            "  NEO4J_DATABASE=neo4j  (or leave unset for default)\n\n"
            "For remote Neo4j:\n"
            "  NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io\n"
            "  NEO4J_USERNAME=neo4j\n"
            "  NEO4J_PASSWORD=your_password\n"
            "  NEO4J_DATABASE=neo4j"
        )
    
    return {
        "uri": uri,
        "username": username,
        "password": password,
        "database": database,
    }


def get_data_paths(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Get standard data file paths.
    
    Session files are stored in the data directory for better organization
    and to keep them with other generated files. Session files are automatically
    set with secure permissions (600 = owner read/write only).
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Session file in data directory (better organization)
    session_file = data_dir / "telegram_parser_session.session"
    
    # Ensure session file has secure permissions if it exists
    if session_file.exists():
        try:
            session_file.chmod(0o600)  # rw------- (owner only)
        except OSError:
            # Permission setting may fail on some systems (e.g., Windows), continue anyway
            pass
    
    return {
        "data_dir": data_dir,
        "messages_file": data_dir / "telegram_messages_extended.jsonl",
        "channels_file": data_dir / "telegram_channels_metadata.jsonl",
        # Parquet files for pandas analysis
        # "messages_parquet": data_dir / "telegram_messages.parquet",
        "messages_parquet": data_dir / "telegram_messages_fresh_full_for_labeling_20260228.parquet",
        "channels_parquet": data_dir / "telegram_channels.parquet",
        "similarity_parquet": data_dir / "message_similarity.parquet",
        "graphml_file": data_dir / "telegram_graph.graphml",
        "session_file": session_file,
    }


def get_session_path(base_dir: Optional[Path] = None, session_name: Optional[str] = None) -> Path:
    """Get path to Telegram session file.
    
    Args:
        base_dir: Base directory (default: current working directory)
        session_name: Custom session name (default: "telegram_parser_session")
    
    Returns:
        Path to session file
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    if session_name is None:
        session_name = "telegram_parser_session"
    
    # Ensure .session extension
    if not session_name.endswith(".session"):
        session_name += ".session"
    
    session_file = data_dir / session_name
    
    # Set secure permissions if file exists
    if session_file.exists():
        try:
            session_file.chmod(0o600)  # rw------- (owner only)
        except OSError:
            pass
    
    return session_file


def clear_session(base_dir: Optional[Path] = None, session_name: Optional[str] = None) -> bool:
    """Clear/delete a Telegram session file.
    
    Useful when session is invalid or expired.
    
    Args:
        base_dir: Base directory (default: current working directory)
        session_name: Session name to clear (default: "telegram_parser_session")
    
    Returns:
        True if session was deleted, False if it didn't exist
    """
    session_file = get_session_path(base_dir, session_name)
    
    if session_file.exists():
        try:
            session_file.unlink()
            return True
        except OSError as e:
            raise OSError(f"Failed to delete session file {session_file}: {e}")
    
    return False

