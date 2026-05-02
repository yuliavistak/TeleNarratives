from pathlib import Path

from dotenv import load_dotenv

import pandas as pd
from openai import OpenAI

from embedding_utils import iter_batches, run_similarity_pipeline

# ---- Config ----
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # TeleNarratives/

MODEL = "text-embedding-3-large"
# MODEL = "text-embedding-3-small"
OUT_CSV = "labeled_messages_openai_embeddings_large_100_2.csv"

MESSAGES_CSV = _PROJECT_ROOT / "data" / "messages_103.csv"
NARRATIVES_CSV = _PROJECT_ROOT / "data" / "Narratives.csv"

NARRATIVE_TEMPLATE = "{sub_narrative}"
MESSAGE_PREFIX_CHARS = None  # e.g. 500 to use only first 500 chars
# USE_CONTEXT = False
# CONTEXT_TEMPLATE = "{date_utc} {channel_username}"

EMBED_BATCH_SIZE = 128
# MESSAGE_BATCH_SIZE = 64
MESSAGE_BATCH_SIZE = 32
THRESHOLD = 0.7  # applied after score_transform

load_dotenv()  # loads OPENAI_API_KEY
client = OpenAI()


def _score_transform(cos_sim: float) -> float:
    # Map cosine similarity from [-1, 1] to [0, 1]
    return (cos_sim + 1.0) / 2.0


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in iter_batches(texts, EMBED_BATCH_SIZE):
        resp = client.embeddings.create(model=MODEL, input=batch)
        data = sorted(resp.data, key=lambda d: d.index)
        embeddings.extend([item.embedding for item in data])
    return embeddings


def main():
    messages = pd.read_csv(MESSAGES_CSV)
    # messages = messages.sample(20, random_state=42)
    narratives = pd.read_csv(NARRATIVES_CSV)

    run_similarity_pipeline(
        messages_df=messages,
        narratives_df=narratives,
        embed_texts=embed_texts,
        out_csv=OUT_CSV,
        narrative_template=NARRATIVE_TEMPLATE,
        message_text_col="text",
        message_prefix_chars=MESSAGE_PREFIX_CHARS,
        # include_context=USE_CONTEXT,
        # context_template=CONTEXT_TEMPLATE,
        threshold=THRESHOLD,
        message_batch_size=MESSAGE_BATCH_SIZE,
        score_transform=_score_transform,
        progress_every=10,
    )


if __name__ == "__main__":
    main()
