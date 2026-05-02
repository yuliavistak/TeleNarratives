"""Utilities for exporting a Neo4j graph into a DGL graph for GNN training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


from disinfograph.gnn.dgl import save_dgl_graph


def _fetch_dataframe(session, query: str) -> pd.DataFrame:
    """Run a Cypher query and return the results as a DataFrame.

    Args:
        session: An open Neo4j session.
        query: Cypher query string to execute.

    Returns:
        A pandas DataFrame with one row per result record.
    """
    result = session.run(query)
    return pd.DataFrame([dict(record) for record in result])


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure the given DataFrame contains the requested columns."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _expand_props_frame(
    df: pd.DataFrame,
    columns: Iterable[str],
    props_col: str = "props",
    extra_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Expand a Neo4j properties map into a normal DataFrame."""
    columns = list(columns)
    records = []
    if df.empty:
        return pd.DataFrame(columns=columns + list(extra_cols or []))

    extra_cols = list(extra_cols or [])
    for row in df.to_dict(orient="records"):
        props = row.get(props_col) or {}
        record = {col: props.get(col) for col in columns}
        for extra_col in extra_cols:
            record[extra_col] = row.get(extra_col)
        records.append(record)

    return pd.DataFrame(records)


def _choose_split_cut(
    labels: np.ndarray,
    time_buckets: np.ndarray,
    target_count: float,
    target_pos_rate: float,
    rng: np.random.Generator,
) -> int:
    """Choose a temporal split point, preferring ISO week boundaries."""
    n_items = len(labels)
    if n_items == 0:
        return 0

    cuts = np.arange(n_items + 1, dtype=np.int64)
    week_boundary = np.zeros(n_items + 1, dtype=bool)
    week_boundary[[0, n_items]] = True
    if n_items > 1:
        week_boundary[np.where(time_buckets[1:] != time_buckets[:-1])[0] + 1] = True

    prefix_pos = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    prefix_rates = np.full(n_items + 1, target_pos_rate, dtype=np.float64)
    non_empty = cuts > 0
    prefix_rates[non_empty] = prefix_pos[non_empty] / cuts[non_empty]

    count_score = np.abs(cuts - target_count)
    label_weight = min(max(n_items * 0.1, 1.0), 10.0)
    label_score = np.abs(prefix_rates - target_pos_rate) * label_weight
    boundary_penalty = np.where(week_boundary, 0.0, 0.25)
    tie_break = rng.random(n_items + 1) * 1e-6
    total_score = count_score + label_score + boundary_penalty + tie_break
    return int(cuts[np.argmin(total_score)])


def _build_stratified_masks(
    labels: pd.Series,
    channel_ids: pd.Series,
    timestamps: pd.Series,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible channel-aware chronological train/val/test masks."""
    if not (
        0.0 <= train_frac <= 1.0
        and 0.0 <= val_frac <= 1.0
        and 0.0 <= test_frac <= 1.0
    ):
        raise ValueError(
            "train_frac, val_frac and test_frac must all be between 0.0 and 1.0."
        )
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")
    if not (len(labels) == len(channel_ids) == len(timestamps)):
        raise ValueError("labels, channel_ids and timestamps must have the same length.")

    rng = np.random.default_rng(random_seed)
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)

    split_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(labels), dtype=np.int64),
            "label": labels.fillna(0).astype(int).to_numpy(),
            "channel_id": pd.Series(channel_ids).to_numpy(),
            "timestamp": pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True),
        }
    )

    if split_df.empty:
        return train_mask, val_mask, test_mask

    valid_timestamps = split_df["timestamp"].dropna()
    fallback_timestamp = (
        valid_timestamps.min()
        if not valid_timestamps.empty
        else pd.Timestamp("1970-01-01", tz="UTC")
    )
    channel_min_timestamp = split_df.groupby("channel_id")["timestamp"].transform("min")
    split_df["timestamp"] = (
        split_df["timestamp"].fillna(channel_min_timestamp).fillna(fallback_timestamp)
    )

    iso_calendar = split_df["timestamp"].dt.isocalendar()
    split_df["iso_year"] = iso_calendar.year.astype(np.int32)
    split_df["iso_week"] = iso_calendar.week.astype(np.int32)
    split_df["time_bucket"] = (
        split_df["iso_year"] * 100 + split_df["iso_week"]
    ).astype(np.int32)
    split_df["tie_break"] = rng.random(len(split_df))

    target_pos_rate = float(split_df["label"].mean()) if len(split_df) else 0.0

    for _, channel_df in split_df.groupby("channel_id", sort=False):
        ordered = channel_df.sort_values(
            by=["time_bucket", "timestamp", "tie_break", "row_idx"],
            kind="stable",
        ).reset_index(drop=True)

        channel_labels = ordered["label"].to_numpy(dtype=np.int64)
        channel_buckets = ordered["time_bucket"].to_numpy(dtype=np.int32)
        channel_indices = ordered["row_idx"].to_numpy(dtype=np.int64)

        train_cut = _choose_split_cut(
            labels=channel_labels,
            time_buckets=channel_buckets,
            target_count=train_frac * len(ordered),
            target_pos_rate=target_pos_rate,
            rng=rng,
        )

        remaining_labels = channel_labels[train_cut:]
        remaining_buckets = channel_buckets[train_cut:]
        remaining_frac = val_frac + test_frac
        val_share_of_remaining = (
            val_frac / remaining_frac if remaining_frac > 0.0 else 0.0
        )
        val_cut_relative = _choose_split_cut(
            labels=remaining_labels,
            time_buckets=remaining_buckets,
            target_count=val_share_of_remaining * len(remaining_labels),
            target_pos_rate=target_pos_rate,
            rng=rng,
        )
        val_cut = train_cut + val_cut_relative

        train_mask[channel_indices[:train_cut]] = True
        val_mask[channel_indices[train_cut:val_cut]] = True
        test_mask[channel_indices[val_cut:]] = True

    return train_mask, val_mask, test_mask


def _build_node_maps(
    messages_df: pd.DataFrame,
    channels_df: pd.DataFrame,
) -> Tuple[Dict, Dict, Any]:
    """Build integer index maps from node identity keys to DGL node indices.

    Args:
        messages_df: DataFrame with ``channel_id`` and ``message_id`` columns.
        channels_df: DataFrame with a ``channel_id`` column.

    Returns:
        A tuple of:
        - *message_map*: dict mapping ``(channel_id, message_id)`` to a DGL
          message node index.
        - *channel_map*: dict mapping ``channel_id`` to a DGL channel node index.
        - *channel_ids*: Sorted unique channel ID array used to build the map.
    """
    message_keys = list(
        messages_df[["channel_id", "message_id"]].itertuples(index=False, name=None)
    )
    message_map = {key: idx for idx, key in enumerate(message_keys)}

    channel_ids = channels_df["channel_id"].dropna().astype(int).sort_values().unique()
    channel_map = {channel_id: idx for idx, channel_id in enumerate(channel_ids)}
    return message_map, channel_map, channel_ids


def _to_edge_arrays(
    df: pd.DataFrame,
    src_cols: Tuple[str, ...],
    tgt_cols: Tuple[str, ...],
    src_map: Dict,
    tgt_map: Dict,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert relationship rows into DGL-compatible source/target arrays."""
    if df.empty:
        return None

    src_values = []
    tgt_values = []
    for row in df.itertuples(index=False):
        src_key = tuple(getattr(row, col) for col in src_cols)
        tgt_key = tuple(getattr(row, col) for col in tgt_cols)
        if len(src_key) == 1:
            src_key = src_key[0]
        if len(tgt_key) == 1:
            tgt_key = tgt_key[0]
        src_idx = src_map.get(src_key)
        tgt_idx = tgt_map.get(tgt_key)
        if src_idx is None or tgt_idx is None:
            continue
        src_values.append(src_idx)
        tgt_values.append(tgt_idx)

    if not src_values:
        return None

    return (
        np.asarray(src_values, dtype=np.int64),
        np.asarray(tgt_values, dtype=np.int64),
    )


def _export_split_samples(
    messages_df: pd.DataFrame,
    labels: pd.Series,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save split membership to CSV files for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_frame = messages_df[["channel_id", "message_id", "date_utc"]].copy()
    split_frame["label"] = labels.fillna(0).astype(int).to_numpy()
    split_frame["date_utc"] = pd.to_datetime(split_frame["date_utc"], errors="coerce")

    split_masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }
    for split_name, split_mask in split_masks.items():
        split_output = output_path.with_name(
            f"{output_path.stem}_{split_name}_samples.csv"
        )
        split_frame.loc[split_mask, ["channel_id", "message_id", "date_utc", "label"]].to_csv(
            split_output,
            index=False,
        )



def load_training_frames_from_neo4j(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch the graph slices needed to build a DGL training graph.
    Returns a dictionary of DataFrames, typically including:
    - "channels": one row per Channel node, with properties expanded into columns
    - "messages": one row per Message node, with properties expanded into columns
    - "forward": one row per FORWARD_FROM relationship, with source/target message IDs
    """

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database, fetch_size=500) as session:
            raw_channels_df = _fetch_dataframe(
                session,
                """
                MATCH (c:Channel)
                RETURN 
                    properties(c) AS props
                """,
            )
            raw_messages_df = _fetch_dataframe(
                session,
                """
                MATCH (m:Message)
                RETURN
                    properties(m) AS props
                """,
            )
            forward_msg_df = _fetch_dataframe(
                session,
                """
                MATCH (src:Message)-[r:FORWARD_FROM]->(dst:Message)
                RETURN
                    src.channel_id AS src_channel_id,
                    src.message_id AS src_message_id,
                    dst.channel_id AS dst_channel_id,
                    dst.message_id AS dst_message_id
                """,
            )
            forward_ch_df = _fetch_dataframe(
                session,
                """
                MATCH (src:Message)-[r:FORWARD_FROM]->(dst:Channel)
                RETURN
                    src.channel_id AS src_channel_id,
                    src.message_id AS src_message_id,
                    dst.channel_id AS dst_channel_id
                """,
            )
            similar_to_df = _fetch_dataframe(
                session,
                """
                MATCH (src:Message)-[r:SIMILAR_TO]->(dst:Message)
                RETURN
                    src.channel_id AS src_channel_id,
                    src.message_id AS src_message_id,
                    dst.channel_id AS dst_channel_id,
                    dst.message_id AS dst_message_id
                """,
            )

    finally:
        driver.close()

    channels_df = _expand_props_frame(
        raw_channels_df,
        props_col= "props",
        columns=["channel_id", "username", "posted_messages", "reliability", "label"],
    ).rename(columns={"label": "channel_label"})

    messages_df = _expand_props_frame(
        raw_messages_df,
        columns=[
            "channel_id",
            "message_id",
            "channel_username",
            "date_utc",
            "sub_narrative_id",
            "sub_narrative",
            "narrative_id",
            "narrative",
            "meta_narrative_id",
            "meta_narrative",
            "label",
            "text",
            "embedding"
        ],
    ).rename(columns={"label": "message_label"})

    frames = {
        "channels": _ensure_columns(
            channels_df,
            ["channel_id", "username", "posted_messages", "reliability", "label"],
        ),
        "messages": _ensure_columns(
            messages_df,
            [
            "channel_id",
            "message_id",
            "channel_username",
            "date_utc",
            "sub_narrative_id",
            "sub_narrative",
            "narrative_id",
            "narrative",
            "meta_narrative_id",
            "meta_narrative",
            "label",
            "text",
            "embedding"
        ],
        ),
        "forward_msg": _ensure_columns(
            forward_msg_df,
            ["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"],
        ),
        "forward_ch": _ensure_columns(
            forward_ch_df,
            ["src_channel_id", "src_message_id", "dst_channel_id"],
        ),
        "similar_to": _ensure_columns(
            similar_to_df,
            ["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"],
        ),
    }

    if frames["messages"].empty:
        raise ValueError("Neo4j export returned no Message nodes to train on.")

    return frames


def build_dgl_graph_from_neo4j(
    output_path: Path,
    target_property: str = "label",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 4242,
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
):
    """
    Build a heterogeneous DGL graph from the contents of Neo4j.
     - Nodes: Channel and Message nodes from Neo4j, with properties expanded into featuresq
     - Edges: "posted" edges from Channel to Message, and "forward_from" edges between Messages
    The resulting graph is saved to disk at the specified output path.
    """
    try:
        import dgl
        import torch
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise ModuleNotFoundError(
            "DGL could not be imported to export the training graph. "
            "This environment likely has an incompatible DGL/PyTorch installation."
        ) from exc

    frames = load_training_frames_from_neo4j(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )
    channels_df = frames["channels"].copy()
    messages_df = frames["messages"].copy()
    forward_msg_df = frames["forward_msg"].copy()
    forward_ch_df = frames["forward_ch"].copy()
    similar_to_df = frames["similar_to"].copy()

    if target_property not in messages_df.columns:
        raise ValueError(
            f"Target property '{target_property}' was not returned from Neo4j. "
            "This exporter currently expects the training target on Message.label."
        )

    channels_df["channel_id"] = pd.to_numeric(channels_df["channel_id"], errors="coerce").astype("Int64")
    channels_df = channels_df.dropna(subset=["channel_id"]).copy()
    channels_df["channel_id"] = channels_df["channel_id"].astype(int)
    channels_df = channels_df.drop_duplicates(subset=["channel_id"]).reset_index(drop=True)

    messages_df["channel_id"] = pd.to_numeric(messages_df["channel_id"], errors="coerce").astype("Int64")
    messages_df["message_id"] = pd.to_numeric(messages_df["message_id"], errors="coerce").astype("Int64")
    messages_df = messages_df.dropna(subset=["channel_id", "message_id"]).copy()
    messages_df["channel_id"] = messages_df["channel_id"].astype(int)
    messages_df["message_id"] = messages_df["message_id"].astype(int)
    messages_df = messages_df.drop_duplicates(subset=["channel_id", "message_id"]).reset_index(drop=True)

    for col in ["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"]:
        forward_msg_df[col] = pd.to_numeric(forward_msg_df[col], errors="coerce").astype("Int64")
    forward_msg_df = forward_msg_df.dropna(subset=["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"]).copy()
    for col in ["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"]:
        forward_msg_df[col] = forward_msg_df[col].astype(int)

    for col in ["src_channel_id", "src_message_id", "dst_channel_id"]:
        forward_ch_df[col] = pd.to_numeric(forward_ch_df[col], errors="coerce").astype("Int64")
    forward_ch_df = forward_ch_df.dropna(subset=["src_channel_id", "src_message_id", "dst_channel_id"]).copy()
    for col in ["src_channel_id", "src_message_id", "dst_channel_id"]:
        forward_ch_df[col] = forward_ch_df[col].astype(int)

    for col in ["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"]:
        similar_to_df[col] = pd.to_numeric(similar_to_df[col], errors="coerce").astype("Int64")
    similar_to_df = similar_to_df.dropna(subset=["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"]).copy()
    for col in ["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"]:
        similar_to_df[col] = similar_to_df[col].astype(int)


    # print(f"Computing text embeddings for {len(channels_df)} channels...")
    # channels_df["embedding"] = compute_text_embeddings(channels_df['username'].fillna("").tolist())
    print('Finished computing channel embeddings.')

    messages_df["channel_id"] = messages_df["channel_id"].astype(int)
    messages_df["message_id"] = messages_df["message_id"].astype(int)
    messages_df["channel_username"] = messages_df["channel_username"].astype(str)

    messages_df["message_label"] = messages_df["message_label"].fillna(0).astype(int)

    messages_df["text_len"] = np.log1p(messages_df["text"].str.len().fillna(0)).astype(np.float32)

    messages_df["date_utc"] = pd.to_datetime(messages_df["date_utc"], errors="coerce")
    messages_df["year"] = (messages_df["date_utc"].dt.year.fillna(0).astype(np.float32))
    messages_df["month"] = (messages_df["date_utc"].dt.month.fillna(0).astype(np.float32))
    messages_df["week"] = (messages_df["date_utc"].dt.isocalendar().week.fillna(0).astype(np.float32))
    messages_df["day"] = (messages_df["date_utc"].dt.dayofweek.fillna(0).astype(np.float32))
    messages_df["hour"] = (messages_df["date_utc"].dt.hour.fillna(0).astype(np.float32))
    # print(f"Computing text embeddings for {len(messages_df)} messages...")

    # messages_df["embedding"] = compute_text_embeddings(messages_df['text'].fillna("").tolist())
    print('Finished computing message embeddings.')

    message_map, channel_map, channel_ids = _build_node_maps(messages_df, channels_df)

    graph_data = {}
    posted_src = np.asarray([channel_map[channel_id] for channel_id in messages_df["channel_id"].tolist()], dtype=np.int64)
    posted_tgt = np.arange(len(messages_df), dtype=np.int64)
    graph_data[("channel", "posted", "message")] = (posted_src, posted_tgt)
    graph_data[("message", "posted_inv", "channel")] = (posted_tgt, posted_src)

    forward_msg_df = forward_msg_df.drop_duplicates().copy()
    if not forward_msg_df.empty:
        fwd_msg_arrays = _to_edge_arrays(
            forward_msg_df,
            ("src_channel_id", "src_message_id"),
            ("dst_channel_id", "dst_message_id"),
            message_map,
            message_map,
        )
        if fwd_msg_arrays is not None:
            graph_data[("message", "forward_from", "message")] = fwd_msg_arrays
            graph_data[("message", "forward_from_inv", "message")] = (
                fwd_msg_arrays[1],
                fwd_msg_arrays[0],
            )

    forward_ch_df = forward_ch_df.drop_duplicates().copy()
    if not forward_ch_df.empty:
        fwd_ch_arrays = _to_edge_arrays(
            forward_ch_df,
            ("src_channel_id", "src_message_id"),
            ("dst_channel_id",),
            message_map,
            channel_map,
        )
        if fwd_ch_arrays is not None:
            graph_data[("message", "forward_from", "channel")] = fwd_ch_arrays
            graph_data[("channel", "forward_from_inv", "message")] = (
                fwd_ch_arrays[1],
                fwd_ch_arrays[0],
            )

    similar_to_df = similar_to_df.drop_duplicates().copy()
    if not similar_to_df.empty:
        sim_arrays = _to_edge_arrays(
            similar_to_df,
            ("src_channel_id", "src_message_id"),
            ("dst_channel_id", "dst_message_id"),
            message_map,
            message_map,
        )
        if sim_arrays is not None:
            graph_data[("message", "similar_to", "message")] = sim_arrays
            graph_data[("message", "similar_to_inv", "message")] = (
                sim_arrays[1],
                sim_arrays[0],
            )

    num_nodes_dict = {
        "channel": len(channel_ids),
        "message": len(messages_df),
    }

    graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    channel_feats = np.stack(
        [
            # channels_df["username"].astype(str),
            channels_df["posted_messages"].to_numpy(dtype=np.float32),

        ],
        axis=1,
    )

    graph.nodes["channel"].data["feat"] = torch.from_numpy(channel_feats)
    graph.nodes["channel"].data["channel_id"] = torch.from_numpy(
        np.asarray(channel_ids, dtype=np.int64)
    )

    emb_list = messages_df['embedding'].tolist()
    emb_dim = next((len(e) for e in emb_list if e is not None), 1024)
    emb_list = [e if e is not None else [0.0] * emb_dim for e in emb_list]
    message_embeddings = np.array(emb_list, dtype=np.float32)
    message_other_features = np.column_stack([
        messages_df["text_len"].to_numpy(dtype=np.float32).reshape(-1,1),
        messages_df["year"].values.reshape(-1,1),
        messages_df["month"].values.reshape(-1,1),
        messages_df["week"].values.reshape(-1,1),
        messages_df["day"].values.reshape(-1,1),
        messages_df["hour"].values.reshape(-1,1),
    ])
    message_feats = np.concatenate([message_embeddings, message_other_features], axis=1)
    graph.nodes["message"].data["feat"] = torch.from_numpy(message_feats.astype(np.float32))
    graph.nodes["message"].data["message_id"] = torch.from_numpy(
        messages_df["message_id"].to_numpy(dtype=np.int64)
    )
    graph.nodes["message"].data["channel_id"] = torch.from_numpy(
        messages_df["channel_id"].to_numpy(dtype=np.int64)
    )

    labels = messages_df[target_property].fillna(0).astype(int)
    graph.nodes["message"].data["label"] = torch.from_numpy(
        labels.to_numpy(dtype=np.int64)
    )



    train_mask, val_mask, test_mask = _build_stratified_masks(
        labels=labels,
        channel_ids=messages_df["channel_id"],
        timestamps=messages_df["date_utc"],
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_seed=random_seed,
    )


    graph.nodes["message"].data["train_mask"] = torch.from_numpy(train_mask)
    graph.nodes["message"].data["val_mask"] = torch.from_numpy(val_mask)
    graph.nodes["message"].data["test_mask"] = torch.from_numpy(test_mask)

    _export_split_samples(
        messages_df=messages_df,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        output_path=output_path,
    )

    save_dgl_graph(graph, output_path)
    return graph
