"""Neo4j data loader — push a NetworkX graph into a Neo4j database."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import pandas as pd
from neo4j import GraphDatabase

from disinfograph.gnn.graph_builder import similarity_ratio


def chunked(iterable, size: int):
    """Yield successive fixed-size chunks from *iterable*.

    Args:
        iterable: Any iterable to chunk.
        size: Maximum number of items per chunk.

    Yields:
        Lists of up to *size* items.
    """
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        if not chunk:
            break
        yield chunk


def _open_session(driver: GraphDatabase.driver, database: Optional[str] = None):
    """Open a Neo4j session, optionally bound to a named database.

    Args:
        driver: An active Neo4j driver instance.
        database: Target database name. Uses the default database when None.

    Returns:
        An open Neo4j session context manager.
    """
    if database:
        return driver.session(database=database)
    return driver.session()


def _run_write(session, query: str, **parameters) -> None:
    """Run a Neo4j write query and fully consume the result."""
    session.run(query, **parameters).consume()


def _sanitize_neo4j_value(value: Any) -> Any:
    """Convert pandas/numpy-ish values to Neo4j-friendly Python scalars."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for item in value:
            item = _sanitize_neo4j_value(item)
            if item is not None:
                cleaned.append(item)
        return cleaned

    return value


def _clean_graph_properties(node_or_edge_data: Dict[str, Any], exclude: Set[str]) -> Dict[str, Any]:
    """Remove graph metadata keys and null values before sending to Neo4j.

    Args:
        node_or_edge_data: Raw attribute dictionary from a NetworkX node or edge.
        exclude: Set of key names to unconditionally skip.

    Returns:
        A cleaned dictionary safe to pass as Neo4j properties.
    """
    properties: Dict[str, Any] = {}
    for key, value in node_or_edge_data.items():
        if key in exclude:
            continue
        value = _sanitize_neo4j_value(value)
        if value is None:
            continue
        properties[key] = value
    return properties


def _get_graph_label(node_data: Dict[str, Any]) -> Optional[str]:
    """Extract the primary Neo4j label from a NetworkX node attribute dict."""
    labels = node_data.get("labels")
    if labels is None:
        return None
    if isinstance(labels, str):
        return labels
    if isinstance(labels, (list, tuple, set)):
        for label in labels:
            if label:
                return str(label)
    return str(labels)


def _print_database_summary(driver: GraphDatabase.driver, database: Optional[str] = None) -> None:
    """Print a concise summary of what is currently stored in Neo4j."""
    with _open_session(driver, database) as session:
        node_total = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        rel_total = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

        node_rows = session.run(
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY label
            """
        ).data()

        rel_rows = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY type
            """
        ).data()

    print("Neo4j database summary:")
    print(f"  Total nodes: {node_total}")
    print(f"  Total relationships: {rel_total}")
    if node_rows:
        print("  Nodes by label:", {row["label"]: row["count"] for row in node_rows})
    if rel_rows:
        print("  Relationships by type:", {row["type"]: row["count"] for row in rel_rows})


def _assert_database_empty(driver: GraphDatabase.driver, database: Optional[str] = None) -> None:
    """Raise if the target database still contains data after a clear step."""
    with _open_session(driver, database) as session:
        node_total = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        rel_total = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

    if node_total != 0 or rel_total != 0:
        raise RuntimeError(
            "Neo4j clear step did not leave the target database empty. "
            f"Remaining nodes: {node_total}, remaining relationships: {rel_total}."
        )


def sync_schema(driver: GraphDatabase.driver, database: Optional[str] = None) -> None:
    """Create uniqueness constraints and indexes in Neo4j.

    Uses the Neo4j 4/5 constraint syntax.  Safe to call on an already-configured
    database — constraints are created only if they do not exist.

    Args:
        driver: An active Neo4j driver instance.
        database: Target database name. Uses the default database when None.
    """
    with _open_session(driver, database) as session:
        # Channel unique by channel_id
        _run_write(
            session,
            """
            CREATE CONSTRAINT channel_id_unique IF NOT EXISTS
            FOR (c:Channel)
            REQUIRE c.channel_id IS UNIQUE
            """
        )

        # Message unique by (channel_id, message_id)
        _run_write(
            session,
            """
            CREATE CONSTRAINT message_identity_unique IF NOT EXISTS
            FOR (m:Message)
            REQUIRE (m.channel_id, m.message_id) IS UNIQUE
            """
        )


def push_channels(
    driver: GraphDatabase.driver,
    channels_df: pd.DataFrame,
    batch_size: int = 100,
    database: Optional[str] = None,
) -> None:
    """Create or merge Channel nodes from a DataFrame.

    Args:
        driver: An active Neo4j driver instance.
        channels_df: DataFrame with at least a ``channel_id`` column and
            optional ``username``, ``title``, and ``label`` columns.
        batch_size: Number of records to send per Cypher transaction.
        database: Target database name. Uses the default database when None.
    """
    records: List[Dict] = []

    for row in channels_df.itertuples(index=False):
        if row.channel_id is None or pd.isna(row.channel_id):
            continue

        rec = {
            "channel_id": int(row.channel_id),
            "username": None if row.username is None or pd.isna(row.username) else str(row.username),
            "title": None if row.title is None or pd.isna(row.title) else str(row.title),
            "label": None if row.label is None or pd.isna(row.label) else str(row.label),
        }
        records.append(rec)

    with _open_session(driver, database) as session:
        for chunk in chunked(records, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MERGE (c:Channel {channel_id: row.channel_id})
                SET c.username = row.username,
                    c.title    = row.title,
                    c.label    = row.label
                """,
                rows=chunk,
            )


def push_messages(
    driver: GraphDatabase.driver,
    messages_df: pd.DataFrame,
    batch_size: int = 200,
    database: Optional[str] = None,
) -> None:
    """Create or merge Message nodes and POSTED relationships.

    Each message row produces a ``Message`` node and a
    ``(:Channel)-[:POSTED]->(:Message)`` relationship.

    Args:
        driver: An active Neo4j driver instance.
        messages_df: DataFrame with ``channel_id``, ``message_id``, and
            optional metadata columns.
        batch_size: Number of records to send per Cypher transaction.
        database: Target database name. Uses the default database when None.
    """
    records: List[Dict] = []

    for row in messages_df.itertuples(index=False):
        if row.channel_id is None or pd.isna(row.channel_id):
            continue
        if row.message_id is None or pd.isna(row.message_id):
            continue

        rec = {
            "channel_id": int(row.channel_id),
            "message_id": int(row.message_id),
            "channel_username": None
            if row.channel_username is None or pd.isna(row.channel_username)
            else str(row.channel_username),
            "date_utc": None if row.date_utc is None or pd.isna(row.date_utc) else str(row.date_utc),
            "views": None if row.views is None or pd.isna(row.views) else int(row.views),
            "forwards": None if row.forwards is None or pd.isna(row.forwards) else int(row.forwards),
            "replies_count": None
            if row.replies_count is None or pd.isna(row.replies_count)
            else int(row.replies_count),
            "has_media": None if row.has_media is None or pd.isna(row.has_media) else bool(row.has_media),
        }
        records.append(rec)

    with _open_session(driver, database) as session:
        for chunk in chunked(records, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MERGE (m:Message {channel_id: row.channel_id, message_id: row.message_id})
                SET m.channel_username = row.channel_username,
                    m.date_utc        = row.date_utc,
                    m.views           = row.views,
                    m.forwards        = row.forwards,
                    m.replies_count   = row.replies_count,
                    m.has_media       = row.has_media
                WITH m, row
                MATCH (c:Channel {channel_id: row.channel_id})
                MERGE (c)-[:POSTED]->(m)
                """,
                rows=chunk,
            )


def push_forward_rels(
    driver: GraphDatabase.driver,
    messages_df: pd.DataFrame,
    batch_size: int = 200,
    database: Optional[str] = None,
) -> None:
    """Create FORWARD_FROM relationships between forwarded and original messages.

    Only rows where ``is_forwarded`` is truthy and all four ID columns are
    non-null are processed.

    Args:
        driver: An active Neo4j driver instance.
        messages_df: DataFrame with ``channel_id``, ``message_id``,
            ``fwd_from_channel_id``, ``fwd_from_message_id``, and
            ``is_forwarded`` columns.
        batch_size: Number of records to send per Cypher transaction.
        database: Target database name. Uses the default database when None.
    """
    records: List[Dict] = []

    for row in messages_df.itertuples(index=False):
        if not bool(row.is_forwarded):
            continue
        if (
            row.channel_id is None
            or pd.isna(row.channel_id)
            or row.message_id is None
            or pd.isna(row.message_id)
        ):
            continue
        if (
            row.fwd_from_channel_id is None
            or pd.isna(row.fwd_from_channel_id)
            or row.fwd_from_message_id is None
            or pd.isna(row.fwd_from_message_id)
        ):
            continue

        rec = {
            "channel_id": int(row.channel_id),
            "message_id": int(row.message_id),
            "src_fwd_from_channel_id": int(row.fwd_from_channel_id),
            "src_fwd_from_message_id": int(row.fwd_from_message_id),
        }
        records.append(rec)

    with _open_session(driver, database) as session:
        for chunk in chunked(records, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MATCH (src:Message {channel_id: row.channel_id, message_id: row.message_id})
                MATCH (dst:Message {
                    channel_id: row.src_fwd_from_channel_id,
                    message_id: row.src_fwd_from_message_id
                })
                MERGE (src)-[:FORWARD_FROM]->(dst)
                """,
                rows=chunk,
            )


def push_similar_rels(
    driver: GraphDatabase.driver,
    messages_df: pd.DataFrame,
    similarity_threshold: float = 0.85,
    min_text_length: int = 20,
    batch_size: int = 200,
    database: Optional[str] = None,
) -> None:
    """Create SIMILAR_TO relationships using pairwise Levenshtein text similarity.

    .. warning::
        This function performs an O(n²) pairwise comparison and is disabled in
        the main pipeline for performance reasons.  Prefer pre-computed
        similarity pairs loaded from a Parquet file instead.

    Args:
        driver: An active Neo4j driver instance.
        messages_df: DataFrame with message data; must include a ``'text'`` column.
        similarity_threshold: Minimum similarity ratio (0.0–1.0) required to
            create a SIMILAR_TO edge.
        min_text_length: Minimum stripped text length for a message to be
            considered for comparison.
        batch_size: Number of pairs to send per Cypher transaction.
        database: Target database name. Uses the default database when None.
    """
    if "text" not in messages_df.columns:
        print("  'text' column not found in messages_df. Skipping SIMILAR_TO relationships.")
        return

    print(
        f"  Computing text similarity "
        f"(threshold: {similarity_threshold}, min length: {min_text_length})..."
    )

    messages_with_text = []
    for row in messages_df.itertuples(index=False):
        text = getattr(row, "text", None)
        if (
            text
            and isinstance(text, str)
            and len(text.strip()) >= min_text_length
            and row.channel_id is not None
            and not pd.isna(row.channel_id)
            and row.message_id is not None
            and not pd.isna(row.message_id)
        ):
            messages_with_text.append({
                "channel_id": int(row.channel_id),
                "message_id": int(row.message_id),
                "text": text.strip(),
            })

    print(f"  Comparing {len(messages_with_text)} messages with sufficient text...")

    similar_pairs = []
    total_comparisons = len(messages_with_text) * (len(messages_with_text) - 1) // 2
    comparison_count = 0

    for i, msg1 in enumerate(messages_with_text):
        for msg2 in messages_with_text[i + 1:]:
            comparison_count += 1
            if comparison_count % 1000 == 0:
                print(f"    Progress: {comparison_count}/{total_comparisons} comparisons...")
            sim = similarity_ratio(msg1["text"], msg2["text"])
            if sim >= similarity_threshold:
                similar_pairs.append({
                    "src_channel_id": msg1["channel_id"],
                    "src_message_id": msg1["message_id"],
                    "dst_channel_id": msg2["channel_id"],
                    "dst_message_id": msg2["message_id"],
                    "similarity_score": sim,
                })

    print(f"  Found {len(similar_pairs)} similar message pairs")

    with _open_session(driver, database) as session:
        for chunk in chunked(similar_pairs, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MATCH (src:Message {channel_id: row.src_channel_id, message_id: row.src_message_id})
                MATCH (dst:Message {channel_id: row.dst_channel_id, message_id: row.dst_message_id})
                MERGE (src)-[r:SIMILAR_TO]->(dst)
                SET r.similarity_score = row.similarity_score
                """,
                rows=chunk,
            )


def push_graph_channels(
    driver: GraphDatabase.driver,
    graph: nx.MultiDiGraph,
    batch_size: int = 100,
    database: Optional[str] = None,
) -> None:
    """Create or update Channel nodes from a NetworkX graph."""
    records: List[Dict[str, Any]] = []

    for _, node_data in graph.nodes(data=True):
        if _get_graph_label(node_data) != "Channel":
            continue

        channel_id = _sanitize_neo4j_value(node_data.get("channel_id"))
        if channel_id is None:
            continue

        records.append(
            {
                "channel_id": int(channel_id),
                "properties": _clean_graph_properties(
                    node_data,
                    exclude={"labels", "node_type", "channel_id"},
                ),
            }
        )

    with _open_session(driver, database) as session:
        for chunk in chunked(records, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MERGE (c:Channel {channel_id: row.channel_id})
                SET c += row.properties
                """,
                rows=chunk,
            )


def push_graph_messages(
    driver: GraphDatabase.driver,
    graph: nx.MultiDiGraph,
    batch_size: int = 200,
    database: Optional[str] = None,
) -> None:
    """Create or update Message nodes from a NetworkX graph."""
    records: List[Dict[str, Any]] = []

    for _, node_data in graph.nodes(data=True):
        if _get_graph_label(node_data) != "Message":
            continue

        channel_id = _sanitize_neo4j_value(node_data.get("channel_id"))
        message_id = _sanitize_neo4j_value(node_data.get("message_id"))
        if channel_id is None or message_id is None:
            continue

        records.append(
            {
                "channel_id": int(channel_id),
                "message_id": int(message_id),
                "properties": _clean_graph_properties(
                    node_data,
                    exclude={"labels", "node_type", "channel_id", "message_id"},
                ),
            }
        )

    with _open_session(driver, database) as session:
        for chunk in chunked(records, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MERGE (m:Message {channel_id: row.channel_id, message_id: row.message_id})
                SET m += row.properties
                """,
                rows=chunk,
            )


def push_graph_named_nodes(
    driver: GraphDatabase.driver,
    graph: nx.MultiDiGraph,
    label: str,
    batch_size: int = 200,
    database: Optional[str] = None,
) -> None:
    """Create or update simple named nodes such as Hashtag and Domain."""
    records: List[Dict[str, Any]] = []

    for _, node_data in graph.nodes(data=True):
        if _get_graph_label(node_data) != label:
            continue

        name = _sanitize_neo4j_value(node_data.get("name"))
        if name is None:
            continue

        records.append(
            {
                "name": str(name),
                "properties": _clean_graph_properties(
                    node_data,
                    exclude={"labels", "node_type", "name"},
                ),
            }
        )

    with _open_session(driver, database) as session:
        for chunk in chunked(records, batch_size):
            _run_write(
                session,
                f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{name: row.name}})
                SET n += row.properties
                """,
                rows=chunk,
            )


def push_graph_relationships(
    driver: GraphDatabase.driver,
    graph: nx.MultiDiGraph,
    batch_size: int = 200,
    database: Optional[str] = None,
) -> None:
    """Create relationships from a NetworkX graph using node identities."""
    posted_rows: List[Dict[str, Any]] = []
    fwd_msg_rows: List[Dict[str, Any]] = []
    fwd_ch_rows: List[Dict[str, Any]] = []
    similar_to_rows: List[Dict[str, Any]] = []

    for src_node, dst_node, _, edge_data in graph.edges(keys=True, data=True):
        rel_type = edge_data.get("type")
        if not rel_type:
            continue

        src_data = graph.nodes[src_node]
        dst_data = graph.nodes[dst_node]

        src_label = _get_graph_label(src_data)
        dst_label = _get_graph_label(dst_data)
        rel_properties = _clean_graph_properties(edge_data, exclude={"type"})

        if rel_type == "POSTED" and src_label == "Channel" and dst_label == "Message":
            src_channel_id = _sanitize_neo4j_value(src_data.get("channel_id"))
            dst_channel_id = _sanitize_neo4j_value(dst_data.get("channel_id"))
            dst_message_id = _sanitize_neo4j_value(dst_data.get("message_id"))
            if None in (src_channel_id, dst_channel_id, dst_message_id):
                continue
            posted_rows.append(
                {
                    "src_channel_id": int(src_channel_id),
                    "dst_channel_id": int(dst_channel_id),
                    "dst_message_id": int(dst_message_id),
                    "properties": rel_properties,
                }
            )
        elif rel_type == "FORWARD_FROM" and src_label == "Message" and dst_label == "Message":
            src_channel_id = _sanitize_neo4j_value(src_data.get("channel_id"))
            src_message_id = _sanitize_neo4j_value(src_data.get("message_id"))
            dst_channel_id = _sanitize_neo4j_value(dst_data.get("channel_id"))
            dst_message_id = _sanitize_neo4j_value(dst_data.get("message_id"))
            if None in (src_channel_id, src_message_id, dst_channel_id, dst_message_id):
                continue
            fwd_msg_rows.append(
                {
                    "src_channel_id": int(src_channel_id),
                    "src_message_id": int(src_message_id),
                    "dst_channel_id": int(dst_channel_id),
                    "dst_message_id": int(dst_message_id),
                    "properties": rel_properties,
                }
            )
        elif rel_type == "FORWARD_FROM" and src_label == "Message" and dst_label == "Channel":
            src_channel_id = _sanitize_neo4j_value(src_data.get("channel_id"))
            src_message_id = _sanitize_neo4j_value(src_data.get("message_id"))
            dst_channel_id = _sanitize_neo4j_value(dst_data.get("channel_id"))
            if None in (src_channel_id, src_message_id, dst_channel_id):
                continue
            fwd_ch_rows.append(
                {
                    "src_channel_id": int(src_channel_id),
                    "src_message_id": int(src_message_id),
                    "dst_channel_id": int(dst_channel_id),
                    "properties": rel_properties,
                }
            )
        elif rel_type == "SIMILAR_TO" and src_label == "Message" and dst_label == "Message":
            src_channel_id = _sanitize_neo4j_value(src_data.get("channel_id"))
            src_message_id = _sanitize_neo4j_value(src_data.get("message_id"))
            dst_channel_id = _sanitize_neo4j_value(dst_data.get("channel_id"))
            dst_message_id = _sanitize_neo4j_value(dst_data.get("message_id"))
            if None in (src_channel_id, src_message_id, dst_channel_id, dst_message_id):
                continue
            similar_to_rows.append(
                {
                    "src_channel_id": int(src_channel_id),
                    "src_message_id": int(src_message_id),
                    "dst_channel_id": int(dst_channel_id),
                    "dst_message_id": int(dst_message_id),
                    "properties": rel_properties,
                }
            )

    with _open_session(driver, database) as session:
        for chunk in chunked(posted_rows, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MATCH (c:Channel {channel_id: row.src_channel_id})
                MATCH (m:Message {channel_id: row.dst_channel_id, message_id: row.dst_message_id})
                MERGE (c)-[r:POSTED]->(m)
                SET r += row.properties
                """,
                rows=chunk,
            )
        for chunk in chunked(fwd_msg_rows, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MATCH (src:Message {channel_id: row.src_channel_id, message_id: row.src_message_id})
                MATCH (dst:Message {channel_id: row.dst_channel_id, message_id: row.dst_message_id})
                MERGE (src)-[r:FORWARD_FROM]->(dst)
                SET r += row.properties
                """,
                rows=chunk,
            )
        for chunk in chunked(fwd_ch_rows, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MATCH (src:Message {channel_id: row.src_channel_id, message_id: row.src_message_id})
                MATCH (dst:Channel {channel_id: row.dst_channel_id})
                MERGE (src)-[r:FORWARD_FROM]->(dst)
                SET r += row.properties
                """,
                rows=chunk,
            )
        for chunk in chunked(similar_to_rows, batch_size):
            _run_write(
                session,
                """
                UNWIND $rows AS row
                MATCH (src:Message {channel_id: row.src_channel_id, message_id: row.src_message_id})
                MATCH (dst:Message {channel_id: row.dst_channel_id, message_id: row.dst_message_id})
                MERGE (src)-[r:SIMILAR_TO]->(dst)
                SET r += row.properties
                """,
                rows=chunk,
            )



def load_graph_to_neo4j(
    graph: nx.MultiDiGraph,
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    clear_database: bool = False,
) -> None:
    """Load a NetworkX graph produced by the graph builder into Neo4j.

    Pushes Channel nodes, Message nodes, and all relationship types
    (POSTED, FORWARD_FROM, SIMILAR_TO) in batched Cypher transactions.

    Args:
        graph: Heterogeneous directed multigraph returned by
            :func:`~disinfograph.gnn.graph_builder.build_graph_for_neo4j`.
        uri: Neo4j Bolt/neo4j URI (e.g. ``'bolt://localhost:7687'``).
        username: Neo4j authentication username.
        password: Neo4j authentication password.
        database: Target database name. Defaults to ``'neo4j'``.
        clear_database: When True, all existing nodes and relationships are
            deleted before the import.  Use with caution.
    """
    print("Loading data from in-memory graph...")
    print(f"Nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

    node_counts: Dict[str, int] = {}
    for _, node_data in graph.nodes(data=True):
        label = _get_graph_label(node_data) or "Unknown"
        node_counts[label] = node_counts.get(label, 0) + 1

    edge_counts: Dict[str, int] = {}
    for _, _, edge_data in graph.edges(data=True):
        rel_type = edge_data.get("type", "UNKNOWN")
        edge_counts[rel_type] = edge_counts.get(rel_type, 0) + 1

    print(f"Node labels: {node_counts}")
    print(f"Relationship types: {edge_counts}")

    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        if clear_database:
            print("Clearing existing database...")
            with _open_session(driver, database) as session:
                _run_write(session, "MATCH (n) DETACH DELETE n")
            _assert_database_empty(driver, database=database)

        print("Syncing schema (constraints)...")
        sync_schema(driver, database=database)

        print("Pushing channels...")
        push_graph_channels(driver, graph, database=database)

        print("Pushing messages...")
        push_graph_messages(driver, graph, database=database)

        print("Pushing relationships...")
        push_graph_relationships(driver, graph, database=database)

        _print_database_summary(driver, database=database)
    finally:
        driver.close()

    print("Done. Graph pushed to Neo4j.")


# -----------------------------------------------------------------------------
# Legacy Parquet-based loader (kept for backward compatibility)
# -----------------------------------------------------------------------------

def load_to_neo4j(
    parquet_paths: Optional[dict],
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    clear_database: bool = False,
    similarity_threshold: float = 0.85,  # Not used - similarity computation disabled
    min_text_length: int = 20,  # Not used - similarity computation disabled
    graph: Optional[nx.MultiDiGraph] = None,
) -> None:
    """Load data from Parquet files or an in-memory graph to Neo4j.
    
    Args:
        parquet_paths: Dictionary with 'channels_parquet' and 'messages_parquet' paths.
            Required when `graph` is not provided.
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        database: Neo4j database name (default: "neo4j")
        clear_database: If True, clear all data before loading
        similarity_threshold: Not used - similarity computation is disabled
        min_text_length: Not used - similarity computation is disabled
        graph: Optional NetworkX graph produced by the graph builder
    """
    if graph is not None:
        if parquet_paths is not None:
            raise ValueError("Provide either `graph` or `parquet_paths`, not both.")
        load_graph_to_neo4j(
            graph=graph,
            uri=uri,
            username=username,
            password=password,
            database=database,
            clear_database=clear_database,
        )
        return

    if parquet_paths is None:
        raise ValueError("`parquet_paths` is required when `graph` is not provided.")

    print(f"Loading data from Parquet files...")
    print(f"  Channels: {parquet_paths['channels_parquet']}")
    print(f"  Messages: {parquet_paths['messages_parquet']}")
    channels_df, messages_df = load_data(parquet_paths)
    print(f"Channels: {len(channels_df)}, messages: {len(messages_df)}")

    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Optional: clear database if you want a fresh import
    if clear_database:
        print("Clearing existing database...")
        with _open_session(driver, database) as session:
            _run_write(session, "MATCH (n) DETACH DELETE n")
        _assert_database_empty(driver, database=database)

    print("Syncing schema (constraints)...")
    sync_schema(driver, database=database)

    print("Pushing channels...")
    push_channels(driver, channels_df, database=database)

    print("Pushing messages and POSTED relationships...")
    push_messages(driver, messages_df, database=database)

    print("Pushing FORWARD_FROM relationships...")
    push_forward_rels(driver, messages_df, database=database)

    # SIMILAR_TO relationships disabled - similarity computation removed for performance
    # print("Pushing SIMILAR_TO relationships (text similarity)...")
    # push_similar_rels(driver, messages_df, similarity_threshold, min_text_length, database=database)

    _print_database_summary(driver, database=database)

    driver.close()
    print("Done. Data pushed to Neo4j.")
