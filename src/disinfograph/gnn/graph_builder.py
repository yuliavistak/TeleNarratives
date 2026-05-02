"""Graph builder module for creating NetworkX graphs from Parquet data."""

from pathlib import Path
from typing import Any, Optional

import networkx as nx
import pandas as pd


# -----------------------------------------------------------------------------
# Helper functions for data cleaning and normalization
# -----------------------------------------------------------------------------

def deduplicate_forwarded_messages(
    df: pd.DataFrame,
    channel_id_col: str = "channel_id",
    message_id_col: str = "message_id",
    fwd_channel_col: str = "fwd_from_channel_id",
    fwd_message_col: str = "fwd_from_message_id",
) -> pd.DataFrame:
    """Drop forwarded messages whose source is present in the same dataset.

    Telegram forward messages share their text with the original source.
    A forwarded copy is identified by non-null ``fwd_from_channel_id`` and
    ``fwd_from_message_id`` pointing to a ``(channel_id, message_id)`` pair
    that already exists in the dataset.  Removing such copies prevents
    text-level train/test leakage caused by the same content appearing in
    multiple channels.

    Multi-level chains are handled: if C → B → A are all present, both B and C
    are removed and only A is kept (B's source is A which is in the dataset;
    C's source is B which is also in the dataset).

    Note: the unique message identifier is ``(channel_id, message_id)`` —
    ``message_id`` alone is not unique across channels.

    Args:
        df: DataFrame containing message rows.
        channel_id_col: Column name for the posting channel ID.
        message_id_col: Column name for the message ID.
        fwd_channel_col: Column name for the forwarded-from channel ID.
        fwd_message_col: Column name for the forwarded-from message ID.

    Returns:
        A new DataFrame with forward copies removed; the input is not modified.
    """
    known_keys = frozenset(
        zip(
            pd.to_numeric(df[channel_id_col], errors="coerce").dropna().astype(int),
            pd.to_numeric(df[message_id_col], errors="coerce").dropna().astype(int),
        )
    )

    fwd_ch = pd.to_numeric(df[fwd_channel_col], errors="coerce")
    fwd_msg = pd.to_numeric(df[fwd_message_col], errors="coerce")
    fwd_valid = fwd_ch.notna() & fwd_msg.notna()

    fwd_source_keys = pd.Series(
        list(zip(fwd_ch.fillna(-1).astype(int), fwd_msg.fillna(-1).astype(int))),
        index=df.index,
    )
    is_copy_of_known = fwd_valid & fwd_source_keys.apply(lambda k: k in known_keys)

    return df[~is_copy_of_known].reset_index(drop=True)


def _to_str_or_none(value: Any) -> Optional[str]:
    """Convert a value to str, returning None for missing values.

    Args:
        value: Any value, including pandas NA and float NaN.

    Returns:
        String representation of *value*, or None if the value is missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _to_int_or_none(value: Any) -> Optional[int]:
    """Convert a value to int, returning None for missing values.

    Args:
        value: Any value, including pandas NA and float NaN.

    Returns:
        Integer representation of *value*, or None if the value is missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def remove_none_attributes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Remove any node, edge, and graph-level attributes with value None.

    GraphML cannot serialise None values, so this helper strips them before
    export.

    Args:
        graph: A NetworkX MultiDiGraph to clean in-place.

    Returns:
        The same graph with all None-valued attributes removed.
    """
    for _, data in graph.nodes(data=True):
        for key in [k for k, v in data.items() if v is None]:
            del data[key]

    for _, _, data in graph.edges(data=True):
        for key in [k for k, v in data.items() if v is None]:
            del data[key]

    for key in [k for k, v in graph.graph.items() if v is None]:
        del graph.graph[key]

    return graph


# -----------------------------------------------------------------------------
# Main Neo4j graph building functions
# -----------------------------------------------------------------------------

def load_parquet_data(parquet_path: Path) -> pd.DataFrame:
    """Load a Parquet dataset for the custom GNN pipeline.

    Reads the file, normalises column names, and coerces known columns to the
    expected dtypes (datetime, nullable integer, or string).

    Args:
        parquet_path: Path to the Parquet file.

    Returns:
        A pandas DataFrame with normalised and typed columns.
    """
    df = pd.read_parquet(parquet_path)
    df.columns = [str(col).strip() for col in df.columns]

    for col in ["date_utc", "src_date_utc", "dst_date_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in [
        "channel_id",
        "message_id",
        "fwd_from_channel_id",
        "fwd_from_message_id",
        "src_channel_id",
        "src_message_id",
        "dst_channel_id",
        "dst_message_id",
        "sub_narrative_id",
        "narrative_id",
        "meta_narrative_id",
        "label",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in [
        "channel_username",
        "text",
        "src_channel_username",
        "src_text",
        "dst_channel_username",
        "dst_text",
        "sub_narrative",
        "narrative",
        "meta_narrative",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string").replace({"<NA>": pd.NA})

    return df


def build_graph_for_neo4j(
    channels_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    semantic_similarity_pairs_df: Optional[pd.DataFrame] = None,
) -> nx.MultiDiGraph:
    """Build a Neo4j-compatible heterogeneous graph from channel and message data.

    Node labels (Neo4j):
      - Channel
      - Message

    Relationship types (Neo4j):
      - (:Channel)-[:POSTED]->(:Message)
      - (:Message)-[:FORWARD_FROM]->(:Message)
      - (:Message)-[:FORWARD_FROM]->(:Channel)
      - (:Message)-[:SIMILAR_TO]->(:Message)

    Args:
        channels_df: DataFrame with channel metadata (channel_id, channel_username,
            messages_posted, positive_ratio, channel_label).
        messages_df: DataFrame with message data (channel_id, message_id,
            fwd_from_channel_id, fwd_from_message_id, and optional label/text columns).
        semantic_similarity_pairs_df: Optional DataFrame with columns
            (src_channel_id, src_message_id, dst_channel_id, dst_message_id).
            Rows whose endpoints are not present in messages_df are skipped.

    Returns:
        A directed multigraph ready for Neo4j import via GraphML.
    """
    graph = nx.MultiDiGraph()

    # -------------------------------------------------------------------------
    # 1) Add Channel nodes
    # -------------------------------------------------------------------------
    for row in channels_df.itertuples(index=False):
        node_id = f"channel:{row.channel_id}"
        graph.add_node(
            node_id,
            labels="Channel",
            node_type="channel",
            channel_id=_to_int_or_none(row.channel_id),
            username=_to_str_or_none(getattr(row, "channel_username", None)),
            posted_messages=_to_int_or_none(getattr(row, "messages_posted", None)),
            reliability=_to_str_or_none(getattr(row, "positive_ratio", None)),
            label=_to_int_or_none(getattr(row, "channel_label", None)),
        )

    # -------------------------------------------------------------------------
    # 2) Add Message nodes and Channel-[:POSTED]->Message edges
    # -------------------------------------------------------------------------
    msg_key_to_node = {}
    for row in messages_df.itertuples(index=False):
        msg_node = f"msg:{row.channel_id}:{row.message_id}"
        msg_key = (row.channel_id, row.message_id)
        graph.add_node(
            msg_node,
            labels="Message",
            node_type="message",
            channel_id=_to_int_or_none(row.channel_id),
            message_id=_to_int_or_none(row.message_id),
            channel_username=_to_str_or_none(getattr(row, "channel_username", None)),
            date_utc=_to_str_or_none(getattr(row, "date_utc", None)),
            sub_narrative_id=_to_str_or_none(getattr(row, "sub_narrative_id", None)),
            sub_narrative=_to_str_or_none(getattr(row, "sub_narrative", None)),
            narrative_id=_to_str_or_none(getattr(row, "narrative_id", None)),
            narrative=_to_str_or_none(getattr(row, "narrative", None)),
            meta_narrative_id=_to_str_or_none(getattr(row, "meta_narrative_id", None)),
            meta_narrative=_to_str_or_none(getattr(row, "meta_narrative", None)),
            label=_to_int_or_none(getattr(row, "label", None)),
            text=_to_str_or_none(getattr(row, "text", None)),
            embedding=row.embedding if hasattr(row, "embedding") else None,
        )

        msg_key_to_node[msg_key] = msg_node

        ch_node = f"channel:{row.channel_id}"
        if ch_node in graph:
            graph.add_edge(ch_node, msg_node, type="POSTED")

    # -------------------------------------------------------------------------
    # 3) Message-[:FORWARD_FROM]->Message edges (if original is in corpus)
    #    Message-[:FORWARD_FROM]->Channel edges (if source channel is in corpus)
    # -------------------------------------------------------------------------
    for row in messages_df.itertuples(index=False):
        fwd_from_channel_id = getattr(row, "fwd_from_channel_id", None)
        fwd_from_message_id = getattr(row, "fwd_from_message_id", None)
        if (
            fwd_from_channel_id is not None
            and not pd.isna(fwd_from_channel_id)
            and fwd_from_message_id is not None
            and not pd.isna(fwd_from_message_id)
        ):
            src = msg_key_to_node.get((row.channel_id, row.message_id))
            dst = msg_key_to_node.get((fwd_from_channel_id, fwd_from_message_id))
            if src and dst:
                graph.add_edge(src, dst, type="FORWARD_FROM")

            if src:
                ch_node = f"channel:{fwd_from_channel_id}"
                if ch_node in graph:
                    graph.add_edge(src, ch_node, type="FORWARD_FROM")

    # -------------------------------------------------------------------------
    # 4) Message-[:SIMILAR_TO]->Message edges from semantic similarity pairs
    # -------------------------------------------------------------------------
    if semantic_similarity_pairs_df is not None:
        for row in semantic_similarity_pairs_df.itertuples(index=False):
            src = msg_key_to_node.get((row.src_channel_id, row.src_message_id))
            dst = msg_key_to_node.get((row.dst_channel_id, row.dst_message_id))
            if src and dst:
                graph.add_edge(src, dst, type="SIMILAR_TO")

    return graph


# -----------------------------------------------------------------------------
# Similarity functions
# -----------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions,
        substitutions) needed to transform *s1* into *s2*.
    """
    if not s1:
        return len(s2) if s2 else 0
    if not s2:
        return len(s1)

    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )

    return dist[rows - 1][cols - 1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Compute a normalised similarity ratio using Levenshtein distance.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        A float in [0.0, 1.0] where 1.0 means the strings are identical and
        0.0 means they share no characters.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    return 1.0 - (levenshtein_distance(s1, s2) / max_len)
