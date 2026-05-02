"""Telegram channel parser module."""
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timezone

import pyarrow.parquet as pq
import pyarrow as pa
from telethon.sync import TelegramClient
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    SessionPasswordNeededError,
    PhoneCodeInvalidError,
)

from .config import get_channel_labels
from .utils import make_json_safe
from .message_utils import message_to_record, channel_to_record
from .parquet_utils import (
    infer_schema_from_batch,
    write_batch_to_parquet,
)
from .date_utils import get_last_n_months_range

# Constants
DEFAULT_BATCH_SIZE = 100
PROGRESS_UPDATE_INTERVAL = 50


class TelegramParser:
    """Service object responsible for ingesting Telegram channel data into Parquet files.

    JSONL files are optional. By default, only Parquet files are written.

    Configuration (API keys, file paths, channels list, etc.) is passed via
    the constructor so you can reuse this service in other scripts or notebooks.
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str,
        messages_parquet: Path,
        channels_parquet: Path,
        channels: list[str],
        channel_labels: Optional[Dict[str, str]] = None,
        messages_per_channel: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_raw_message: bool = True,
        fetch_reply_parent: bool = True,
        output_messages_file: Optional[Path] = None,
        output_channels_file: Optional[Path] = None,
    ) -> None:
        """Initialize TelegramParser.

        Args:
            api_id: Telegram API ID
            api_hash: Telegram API hash
            session_name: Path to session file
            messages_parquet: Path to output messages Parquet file
            channels_parquet: Path to output channels Parquet file
            channels: List of channel usernames to parse
            channel_labels: Optional dict mapping channel usernames to labels
            messages_per_channel: Optional limit on number of messages per channel.
                IGNORED (date-based filtering is used by default).
                Default behavior: fetch messages from last 12 months.
            start_date: Optional datetime to start fetching from (e.g., 6 months ago).
                If None, defaults to 12 months ago.
                Date-based filtering is always used (takes PRIORITY over messages_per_channel).
            end_date: Optional datetime to stop fetching at (default: now).
                Only used if start_date is provided.
            include_raw_message: Whether to include full raw message data
            fetch_reply_parent: Whether to fetch reply parent metadata
            output_messages_file: Optional path to JSONL file for messages
            output_channels_file: Optional path to JSONL file for channels
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.messages_parquet = messages_parquet
        self.channels_parquet = channels_parquet
        self.channels = channels
        self.channel_labels = channel_labels or get_channel_labels()

        # Date-based filtering (default: last 12 months)
        # If start_date is not provided, default to 12 months ago
        if start_date is not None:
            # User provided explicit start_date
            if start_date.tzinfo is None:
                self.start_date = start_date.replace(tzinfo=timezone.utc)
            else:
                self.start_date = start_date
        else:
            # Default: 12 months ago (date-based filtering by default)
            default_start, default_end = get_last_n_months_range(12)
            self.start_date = default_start

        if end_date is not None:
            if end_date.tzinfo is None:
                self.end_date = end_date.replace(tzinfo=timezone.utc)
            else:
                self.end_date = end_date
        else:
            self.end_date = datetime.now(timezone.utc)

        # Message limit: Date-based filtering is always used (start_date is always set)
        # Since we default to date-based (12 months ago), messages_per_channel is always None
        self.messages_per_channel = None  # Fetch all messages in date range

        self.include_raw_message = include_raw_message
        self.fetch_reply_parent = fetch_reply_parent
        # JSONL files are optional
        self.output_messages_file = output_messages_file
        self.output_channels_file = output_channels_file

    def run(self) -> None:
        """Run full ingestion: fetch from Telegram, write incrementally to Parquet.

        Handles session management, authentication, and connection errors gracefully.
        """
        # Ensure session file has secure permissions (owner read/write only)
        session_path = Path(self.session_name)
        if session_path.exists():
            try:
                session_path.chmod(0o600)  # rw------- (owner only)
            except OSError:
                # Permission setting may fail on some systems, continue anyway
                pass

        # Create client and manage connection manually for better control
        client = TelegramClient(self.session_name, self.api_id, self.api_hash)

        try:
            # Connect and authenticate
            self._connect_and_authenticate(client, session_path)

            channels_seen: set[int] = set()

            # Optional JSONL file handles (only if paths provided)
            f_msg_out = None
            f_ch_out = None
            if self.output_messages_file:
                f_msg_out = self.output_messages_file.open("w", encoding="utf-8")
            if self.output_channels_file:
                f_ch_out = self.output_channels_file.open("w", encoding="utf-8")

            # Initialize Parquet writers for incremental writing
            messages_writer = None
            channels_writer = None
            messages_schema = None
            channels_schema = None

            # Batch buffers for incremental writing
            messages_batch: List[Dict] = []
            channels_batch: List[Dict] = []

            # Track totals
            total_messages = 0
            total_channels = 0

            try:
                for channel_username in self.channels:
                    print(f"\n=== Fetching messages from @{channel_username} ===")

                    try:
                        entity = client.get_entity(channel_username)
                    except Exception as e:
                        print(f"Failed to get entity for {channel_username}: {e}")
                        continue

                    # Save channel metadata once per channel_id
                    ch_id = getattr(entity, "id", None)
                    if ch_id is not None and ch_id not in channels_seen:
                        ch_record = channel_to_record(entity)
                        safe_ch_record = make_json_safe(ch_record)

                        # Add weak label based on username
                        # Normalize the username for label lookup
                        username = ch_record.get("username")
                        username_norm = username.lower() if username else None
                        safe_ch_record["label"] = self.channel_labels.get(username_norm)

                        # Optionally write to JSONL (if file handle provided)
                        if f_ch_out:
                            f_ch_out.write(json.dumps(safe_ch_record, ensure_ascii=False) + "\n")

                        # Add to batch and write immediately (channels are few)
                        channels_batch.append(safe_ch_record)
                        # Initialize schema and writer on first channel
                        if channels_schema is None:
                            channels_schema = self._get_channels_schema(channels_batch)
                            channels_writer = self._init_channels_writer(channels_writer, channels_schema)
                        # Write the channel immediately
                        self._write_channels_batch(channels_batch, channels_writer)
                        channels_batch = []  # Clear after writing
                        total_channels += 1

                        channels_seen.add(ch_id)

                    count = 0

                    # Prepare iter_messages parameters
                    # Date-based filtering is always used (start_date defaults to 12 months ago)
                    iter_kwargs = {}
                    # DATE-BASED FILTERING (always used, defaults to last 12 months)
                    # Fetch messages from start_date to end_date
                    # Use offset_date to start from end_date and iterate backwards
                    # Telethon's iter_messages with offset_date starts from that date and goes backwards
                    iter_kwargs['offset_date'] = self.end_date
                    iter_kwargs['reverse'] = False  # Get newest first (default)
                    # No limit - fetch all messages in date range
                    print(f"  📅 Date-based: Fetching messages from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")

                    for message in client.iter_messages(entity, **iter_kwargs):
                        if not message:
                            continue

                        # Filter by date range (start_date is always set, defaults to 12 months ago)
                        # Skip messages outside the date range
                        if message.date is None:
                            continue

                        msg_date = message.date
                        # message.date from Telethon is timezone-aware (UTC)
                        # Make timezone-aware comparison if needed
                        if msg_date.tzinfo is None:
                            # If message date is naive, assume UTC
                            msg_date = msg_date.replace(tzinfo=timezone.utc)

                        # Compare dates (both should be timezone-aware)
                        if msg_date < self.start_date:
                            # We've gone past the start date, stop fetching
                            print(f"  Reached start date limit ({self.start_date.strftime('%Y-%m-%d')}), stopping")
                            break
                        if msg_date > self.end_date:
                            # Message is too new, skip it (shouldn't happen with offset_date, but check anyway)
                            continue

                        # Convert message to record (media parsing is disabled for performance)
                        record = message_to_record(
                            message, channel_username, client, entity,
                            self.include_raw_message, self.fetch_reply_parent
                        )
                        safe_record = make_json_safe(record)

                        # Optionally write to JSONL (if file handle provided)
                        if f_msg_out:
                            f_msg_out.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

                        # Prepare Parquet record (create a clean record without nested JSON)
                        parquet_record = safe_record.copy()
                        # Keep nested data as JSON strings for Parquet (pandas can handle this)
                        if parquet_record.get("reply_parent_basic"):
                            parquet_record["reply_parent_basic"] = json.dumps(
                                parquet_record["reply_parent_basic"], ensure_ascii=False
                            )
                        if parquet_record.get("fwd_from_raw"):
                            parquet_record["fwd_from_raw"] = json.dumps(
                                parquet_record["fwd_from_raw"], ensure_ascii=False
                            )
                        if parquet_record.get("raw_message"):
                            parquet_record["raw_message"] = json.dumps(
                                parquet_record["raw_message"], ensure_ascii=False
                            )

                        messages_batch.append(parquet_record)
                        total_messages += 1

                        # Write batch to Parquet when batch size is reached
                        if len(messages_batch) >= DEFAULT_BATCH_SIZE:
                            # Initialize schema and writer on first batch
                            if messages_schema is None:
                                messages_schema = self._get_messages_schema(messages_batch)
                                messages_writer = self._init_messages_writer(messages_writer, messages_schema)
                            if messages_writer is not None:
                                self._write_messages_batch(messages_batch, messages_writer, messages_schema)
                                messages_batch = []  # Clear after writing
                                print(f"  ✓ Written {total_messages} messages to Parquet so far...")

                        count += 1
                        if count % PROGRESS_UPDATE_INTERVAL == 0:
                            print(f"  Collected {count} messages from @{channel_username}")

                    print(f"Done: collected {count} messages from @{channel_username}")

                    # Write batch after each channel (ensures progress even with small batches)
                    if messages_batch and len(messages_batch) > 0:
                        if messages_schema is None:
                            messages_schema = self._get_messages_schema(messages_batch)
                            messages_writer = self._init_messages_writer(messages_writer, messages_schema)
                        if messages_writer is not None:
                            self._write_messages_batch(messages_batch, messages_writer, messages_schema)
                            print(f"  ✓ Written batch of {len(messages_batch)} messages to Parquet")
                            messages_batch = []  # Clear after writing

                # Write any remaining batches at the end
                if messages_batch:
                    if messages_schema is None:
                        messages_schema = self._get_messages_schema(messages_batch)
                        messages_writer = self._init_messages_writer(messages_writer, messages_schema)
                    if messages_writer is not None:
                        self._write_messages_batch(messages_batch, messages_writer, messages_schema)
                        print(f"  ✓ Written final batch of {len(messages_batch)} messages to Parquet")
                    messages_batch = []

                if channels_batch:
                    if channels_schema is None:
                        channels_schema = self._get_channels_schema(channels_batch)
                        channels_writer = self._init_channels_writer(channels_writer, channels_schema)
                    self._write_channels_batch(channels_batch, channels_writer)
                    channels_batch = []

            finally:
                # Close Parquet writers
                if messages_writer:
                    messages_writer.close()
                if channels_writer:
                    channels_writer.close()

                # Close JSONL files if they were opened
                if f_msg_out:
                    f_msg_out.close()
                if f_ch_out:
                    f_ch_out.close()

            print(f"\n✓ All data saved to Parquet files")
            print(f"  - Channels: {total_channels}")
            print(f"  - Messages: {total_messages}")
            if self.output_messages_file or self.output_channels_file:
                print(f"  (JSONL files also created)")

        except AuthKeyUnregisteredError:
            print(f"\n❌ Session expired or invalid")
            print(f"   The session file at {session_path} is no longer valid.")
            print("   Please delete it and run the parser again to re-authenticate:")
            print(f"   rm {session_path}")
            raise
        except FloodWaitError as e:
            print(f"\n⚠️  Rate limit: Please wait {e.seconds} seconds before trying again")
            print(f"   Telegram has rate-limited your requests.")
            raise
        except SessionPasswordNeededError:
            # This should be handled in the auth flow above, but just in case
            print(f"\n❌ 2FA password required but not provided")
            raise
        except Exception as e:
            # Handle other authentication and connection errors gracefully
            error_msg = str(e).lower()
            if "auth" in error_msg or "unauthorized" in error_msg or "session" in error_msg:
                print(f"\n❌ Authentication error: {e}")
                print("   This usually means:")
                print("   1. Your session file is invalid or expired")
                print("   2. You need to re-authenticate")
                print(f"   Try deleting the session file: {session_path}")
                print("   Then run the parser again to authenticate.")
            elif "connection" in error_msg or "network" in error_msg or "disconnected" in error_msg:
                print(f"\n❌ Connection error: {e}")
                print("   Please check your internet connection and try again.")
                print("   If the error persists, try deleting the session file and re-authenticating.")
            else:
                print(f"\n❌ Error during parsing: {e}")
                raise

        finally:
            # Always disconnect client properly
            try:
                if client.is_connected():
                    client.disconnect()
            except Exception:
                pass  # Ignore errors during cleanup

    def _connect_and_authenticate(self, client: TelegramClient, session_path: Path) -> None:
        """Connect to Telegram and handle authentication if needed.

        Args:
            client: TelegramClient instance
            session_path: Path to session file (for error messages)

        Raises:
            AuthKeyUnregisteredError: If session is invalid
            PhoneCodeInvalidError: If verification code is invalid
            SessionPasswordNeededError: If 2FA is required but not provided
        """
        # Connect to Telegram (this must be done before any operations)
        if not client.is_connected():
            client.connect()

        # Handle authentication if needed
        if not client.is_user_authorized():
            print("⚠️  Session not authorized. Starting authentication...")
            try:
                # Prompt for phone number
                phone = input("Enter your phone number (with country code, e.g., +1234567890): ")
                client.send_code_request(phone)

                code = input("Enter the verification code you received: ")
                try:
                    client.sign_in(phone, code)
                except SessionPasswordNeededError:
                    # 2FA enabled, need password
                    password = input("Enter your 2FA password: ")
                    client.sign_in(password=password)

                print("✓ Authentication successful")
            except PhoneCodeInvalidError:
                print("❌ Invalid verification code. Please try again.")
                raise
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                raise

        # Ensure we're still connected after authentication
        if not client.is_connected():
            client.connect()

        print("✓ Connected to Telegram")

    def _get_messages_schema(self, sample_batch: List[Dict]) -> Optional[pa.Schema]:
        """Infer Parquet schema from a sample batch of messages.

        The schema is made nullable for all columns to handle missing fields
        in subsequent batches.

        Args:
            sample_batch: List of message dictionaries

        Returns:
            PyArrow Schema or None if batch is empty
        """
        return infer_schema_from_batch(sample_batch, make_nullable=True)

    def _get_channels_schema(self, sample_batch: List[Dict]) -> Optional[pa.Schema]:
        """Infer Parquet schema from a sample batch of channels.

        Args:
            sample_batch: List of channel dictionaries

        Returns:
            PyArrow Schema or None if batch is empty
        """
        return infer_schema_from_batch(sample_batch, make_nullable=False)

    def _init_messages_writer(
        self,
        writer: Optional[pq.ParquetWriter],
        schema: Optional[pa.Schema],
    ) -> Optional[pq.ParquetWriter]:
        """Initialize or return existing Parquet writer for messages."""
        if writer is None and schema is not None:
            # Delete existing file if it exists (start fresh)
            if self.messages_parquet.exists():
                self.messages_parquet.unlink()
            try:
                writer = pq.ParquetWriter(self.messages_parquet, schema)
                print(f"  ✓ Initialized Parquet writer for messages")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not initialize Parquet writer: {e}")
                return None
        return writer

    def _init_channels_writer(
        self,
        writer: Optional[pq.ParquetWriter],
        schema: Optional[pa.Schema],
    ) -> Optional[pq.ParquetWriter]:
        """Initialize or return existing Parquet writer for channels."""
        if writer is None and schema is not None:
            # Delete existing file if it exists (start fresh)
            if self.channels_parquet.exists():
                self.channels_parquet.unlink()
            writer = pq.ParquetWriter(self.channels_parquet, schema)
        return writer

    def _write_messages_batch(
        self,
        batch: List[Dict],
        writer: pq.ParquetWriter,
        schema: Optional[pa.Schema] = None,
    ) -> None:
        """Write a batch of messages to Parquet file.

        Args:
            batch: List of message dictionaries
            writer: ParquetWriter instance
            schema: Optional schema to enforce (ensures consistency across batches)
        """
        try:
            write_batch_to_parquet(batch, writer, schema)
        except Exception as e:
            print(f"  ⚠️  Warning: Error writing messages batch: {e}")
            raise

    def _write_channels_batch(
        self,
        batch: List[Dict],
        writer: pq.ParquetWriter,
    ) -> None:
        """Write a batch of channels to Parquet file.

        Args:
            batch: List of channel dictionaries
            writer: ParquetWriter instance
        """
        if not batch or writer is None:
            return
        write_batch_to_parquet(batch, writer, schema=None)

