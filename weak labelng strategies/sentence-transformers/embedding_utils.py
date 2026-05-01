from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _sanitize(matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all():
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = _sanitize(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(~np.isfinite(norms) | (norms == 0), 1.0, norms)
    matrix = matrix / norms
    return _sanitize(matrix)


def build_candidate_texts(
    narratives_df: pd.DataFrame,
    template: str,
) -> Tuple[List[dict], List[str]]:
    candidates: List[dict] = []
    texts: List[str] = []
    for row in narratives_df.itertuples():
        candidate = {
            "narrative_id": row.narrative_id,
            "sub_narrative_id": row.sub_narrative_id,
            "narrative": "" if pd.isna(row.narrative) else str(row.narrative),
            "sub_narrative": ""
            if pd.isna(row.sub_narrative)
            else str(row.sub_narrative),
        }
        text = template.format(
            narrative=candidate["narrative"],
            sub_narrative=candidate["sub_narrative"],
            narrative_id=candidate["narrative_id"],
            sub_narrative_id=candidate["sub_narrative_id"],
        )
        candidates.append(candidate)
        texts.append(text)
    return candidates, texts


def build_message_texts(
    messages_df: pd.DataFrame,
    text_col: str = "text",
    prefix_chars: int | None = None,
    # include_context: bool = False,
    # context_template: str = "{date_utc} {channel_username}",
) -> List[str]:
    texts: List[str] = []
    for row in messages_df.itertuples():
        raw = getattr(row, text_col, "")
        text = "" if pd.isna(raw) else str(raw)
        if prefix_chars:
            text = text[:prefix_chars]
        # if include_context:
        #     context = context_template.format(**row._asdict())
        #     text = f"{text}\n\nContext: {context}"
        texts.append(text)
    return texts


def run_similarity_pipeline(
    *,
    messages_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
    embed_texts: Callable[[List[str]], List[List[float]]],
    out_csv: str,
    narrative_template: str = "{sub_narrative}",
    message_text_col: str = "text",
    message_prefix_chars: int | None = None,
    # include_context: bool = False,
    # context_template: str = "{date_utc} {channel_username}",
    threshold: float = 0.75,
    message_batch_size: int = 64,
    score_transform: Callable[[float], float] | None = None,
    progress_every: int = 10,
) -> pd.DataFrame:
    candidates, candidate_texts = build_candidate_texts(
        narratives_df, narrative_template
    )
    candidate_embeddings = normalize_rows(embed_texts(candidate_texts))

    message_ids = messages_df["message_id"].tolist()
    message_texts = build_message_texts(
        messages_df,
        text_col=message_text_col,
        prefix_chars=message_prefix_chars,
        # include_context=include_context,
        # context_template=context_template,
    )

    outputs: List[dict] = []
    total = len(message_texts)
    for start in range(0, total, message_batch_size):
        end = min(start + message_batch_size, total)
        batch_texts = message_texts[start:end]
        batch_ids = message_ids[start:end]

        batch_embeddings = normalize_rows(embed_texts(batch_texts))
        sims = batch_embeddings @ candidate_embeddings.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(len(batch_texts)), best_idx]

        for msg_id, idx, sim in zip(batch_ids, best_idx, best_sim):
            sim = float(sim)
            confidence = score_transform(sim) if score_transform else sim
            candidate = candidates[int(idx)]
            if confidence < threshold:
                outputs.append(
                    {
                        "message_id": msg_id,
                        "narrative_id": None,
                        "sub_narrative_id": None,
                        "confidence": 0.0,
                        "reason": (
                            f"no_label (max_score={confidence:.4f} < {threshold})"
                        ),
                    }
                )
            else:
                outputs.append(
                    {
                        "message_id": msg_id,
                        "narrative_id": candidate["narrative_id"],
                        "sub_narrative_id": candidate["sub_narrative_id"],
                        "confidence": float(confidence),
                        "reason": (
                            "top_label="
                            f"{candidate['sub_narrative']} score={confidence:.4f}"
                        ),
                    }
                )

        if progress_every:
            crossed = (start // progress_every) != (end // progress_every)
            if crossed or end == total:
                print(f"Processed {end}/{total}")

    out = pd.DataFrame(outputs)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return out
