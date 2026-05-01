import pandas as pd
import torch
from transformers import pipeline

# ---- Config ----
MODEL = "joeddav/xlm-roberta-large-xnli"
OUT_CSV = "labeled_messages_xlmroberta_100.csv"

MESSAGES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/notebooks/messages_103.csv"
NARRATIVES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/data/Narratives.csv"

HYPOTHESIS_TEMPLATE = "This message expresses the following narrative: {}."
LABEL_BATCH_SIZE = 32
THRESHOLD = 0.97
USE_CONTEXT = True


# ---- Helpers ----
def add_context(df):
    df = df.copy()
    df["context"] = df.apply(
        lambda r: f"{r.date_utc} {r.channel_username}",
        axis=1,
    )
    return df


def load_candidates(narr_df):
    labels = []
    label_to_ids = {}
    for r in narr_df.itertuples():
        label = str(r.sub_narrative)
        if label in label_to_ids:
            raise ValueError(
                "Duplicate sub_narrative text found. "
                "Please deduplicate or include IDs in labels."
            )
        labels.append(label)
        label_to_ids[label] = (str(r.narrative_id), str(r.sub_narrative_id))
    return labels, label_to_ids


def iter_chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def resolve_device():
    if torch.cuda.is_available():
        return 0
    return -1


def score_best_label(zs_pipe, text, labels):
    best_label = None
    best_score = -1.0
    for chunk in iter_chunks(labels, LABEL_BATCH_SIZE):
        result = zs_pipe(
            text,
            candidate_labels=chunk,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
            truncation=True,
        )
        for label, score in zip(result["labels"], result["scores"]):
            score = float(score)
            if score > best_score:
                best_score = score
                best_label = label
    return best_label, best_score


def main():
    messages = pd.read_csv(MESSAGES_CSV)
    # messages = messages.sample(10, random_state=42)
    narratives = pd.read_csv(NARRATIVES_CSV)

    messages = add_context(messages)
    labels, label_to_ids = load_candidates(narratives)

    device = resolve_device()
    zs_pipe = pipeline("zero-shot-classification", model=MODEL, device=device)

    outputs = []
    total = len(messages)
    for idx, row in enumerate(messages.itertuples(), start=1):
        message_text = "" if pd.isna(row.text) else str(row.text)
        if USE_CONTEXT:
            text = f"{message_text}\n\nContext: {row.context}"
        else:
            text = message_text

        label, score = score_best_label(zs_pipe, text, labels)
        if label is None or score < THRESHOLD:
            outputs.append(
                {
                    "message_id": row.message_id,
                    "best_label": label,
                    "score": score,
                    "narrative_id": None,
                    "sub_narrative_id": None,
                    "confidence": 0.0,
                    "reason": f"no_label (max_score={score:.4f} < {THRESHOLD})",
                }
            )
        else:
            narrative_id, sub_narrative_id = label_to_ids[label]
            outputs.append(
                {
                    "message_id": row.message_id,
                    "best_label": label,
                    "score": score,
                    "narrative_id": narrative_id,
                    "sub_narrative_id": sub_narrative_id,
                    "confidence": score,
                    "reason": f"top_label={label} score={score:.4f}",
                }
            )

        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    out = pd.DataFrame(outputs)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
