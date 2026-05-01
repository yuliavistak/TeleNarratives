import pandas as pd

# ---- Config (edit these only) ----
# DATASETS = {
#     "gpt": "labeled_messages_gpt_100.csv",
#     "gemini": "labeled_messages_gemini_100.csv",
#     "claude": "labeled_messages_claude_100.csv",
# }
# # DOMINANT = "gemini"  # must be one of the keys in DATASETS



DATASETS = {
    "gpt": "labeled_messages_gpt_100.csv",
    "gpt41": "labeled_messages_gpt_4.1_100.csv",
    "gemini": "labeled_messages_gemini_100.csv",
}
DOMINANT = "gemini"

OUT_CSV = "labeled_messages_ensemble_vote_3.csv"

ID_COL = "message_id"
# ---- Helpers ----
def _norm_id(x):
    if pd.isna(x):
        return None
    return str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x)


def _as_label(row, prefix):
    n_id = _norm_id(row[f"{prefix}_narrative_id"])
    s_id = _norm_id(row[f"{prefix}_sub_narrative_id"])
    if n_id is None or s_id is None:
        return None
    return (n_id, s_id)


def _majority_vote(labels):
    counts = {}
    for item in labels:
        counts[item] = counts.get(item, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0], best[1]


def _choose_sub_id(labels_by_model, narrative_id, preference):
    for model in preference:
        label = labels_by_model.get(model)
        if label and label[0] == narrative_id:
            return label[1]
    return None


def main():
    models = list(DATASETS.keys())
    if len(models) != 3:
        raise ValueError("DATASETS must contain exactly 3 entries.")
    if DOMINANT not in DATASETS:
        raise ValueError("DOMINANT must be one of the DATASETS keys.")

    frames = []
    for model in models:
        path = DATASETS[model]
        df = pd.read_csv(path)
        df = df[[ID_COL, "narrative_id", "sub_narrative_id"]].rename(
            columns={
                "narrative_id": f"{model}_narrative_id",
                "sub_narrative_id": f"{model}_sub_narrative_id",
            }
        )
        frames.append(df)

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=ID_COL, how="inner")

    print(f"Merged rows: {len(df)}")

    preference = [DOMINANT] + [m for m in models if m != DOMINANT]

    outputs = []
    for row in df.itertuples(index=False):
        row = row._asdict()
        labels_by_model = {m: _as_label(row, m) for m in models}

        # Binary vote: any label != None
        binary_votes = [labels_by_model[m] is not None for m in models]
        positive_votes = sum(binary_votes)

        # If majority says "no narrative"
        if positive_votes < 2:
            outputs.append(
                {
                    ID_COL: row[ID_COL],
                    "narrative_id": None,
                    "sub_narrative_id": None,
                    "confidence": 0.0,
                    "reason": f"binary_vote={positive_votes}/3 -> NONE",
                }
            )
            continue

        # Narrative vote (use narrative_id only)
        narrative_ids = [
            labels_by_model[m][0]
            for m in models
            if labels_by_model[m] is not None
        ]

        # If all three narrative IDs are different, take DOMINANT
        if len(narrative_ids) == 3 and len(set(narrative_ids)) == 3:
            chosen_label = labels_by_model.get(DOMINANT)
            if chosen_label is None:
                # Fallback to first available label
                chosen_label = next(label for label in labels_by_model.values() if label is not None)
            narrative_id = chosen_label[0]
            sub_narrative_id = chosen_label[1]
            vote_share = 1 / 3
            reason = f"all_diff -> {DOMINANT}"
        else:
            narrative_id, count = _majority_vote(narrative_ids)
            sub_narrative_id = _choose_sub_id(labels_by_model, narrative_id, preference)
            vote_share = count / 3
            reason = f"majority_narrative={count}/3"

        outputs.append(
            {
                ID_COL: row[ID_COL],
                "narrative_id": narrative_id,
                "sub_narrative_id": sub_narrative_id,
                "confidence": vote_share,
                "reason": reason,
            }
        )

    out = pd.DataFrame(outputs)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
