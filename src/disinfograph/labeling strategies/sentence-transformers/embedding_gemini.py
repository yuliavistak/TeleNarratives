from pathlib import Path

from dotenv import load_dotenv

import pandas as pd
from google import genai

from embedding_utils import iter_batches, run_similarity_pipeline

# ---- Config ----
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # TeleNarratives/

MODEL = "text-embedding-004"
OUT_CSV = "labeled_messages_gemini_embeddings.csv"

MESSAGES_CSV = _PROJECT_ROOT / "data" / "messages_103.csv"
NARRATIVES_CSV = _PROJECT_ROOT / "data" / "Narratives.csv"

NARRATIVE_TEMPLATE = "{sub_narrative}"
MESSAGE_PREFIX_CHARS = None
USE_CONTEXT = False
CONTEXT_TEMPLATE = "{date_utc} {channel_username}"

EMBED_BATCH_SIZE = 128
MESSAGE_BATCH_SIZE = 64
THRESHOLD = 0.75  # applied after score_transform

load_dotenv()  # loads GEMINI_API_KEY
client = genai.Client()


def _score_transform(cos_sim: float) -> float:
    return (cos_sim + 1.0) / 2.0


def _extract_embedding_values(obj):
    if hasattr(obj, "values"):
        return obj.values
    if hasattr(obj, "embedding"):
        return obj.embedding
    if isinstance(obj, dict):
        if "values" in obj:
            return obj["values"]
        if "embedding" in obj:
            return obj["embedding"]
    return obj


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in iter_batches(texts, EMBED_BATCH_SIZE):
        resp = client.models.embed_content(model=MODEL, contents=batch)
        if hasattr(resp, "embeddings"):
            embeddings.extend([_extract_embedding_values(e) for e in resp.embeddings])
        elif hasattr(resp, "embedding"):
            embeddings.append(_extract_embedding_values(resp.embedding))
        elif isinstance(resp, dict) and "embeddings" in resp:
            embeddings.extend([_extract_embedding_values(e) for e in resp["embeddings"]])
        else:
            raise ValueError("Unexpected Gemini embedding response shape")
    return embeddings


def main():
    messages = pd.read_csv(MESSAGES_CSV)
    narratives = pd.read_csv(NARRATIVES_CSV)

    run_similarity_pipeline(
        messages_df=messages,
        narratives_df=narratives,
        embed_texts=embed_texts,
        out_csv=OUT_CSV,
        narrative_template=NARRATIVE_TEMPLATE,
        message_text_col="text",
        message_prefix_chars=MESSAGE_PREFIX_CHARS,
        include_context=USE_CONTEXT,
        context_template=CONTEXT_TEMPLATE,
        threshold=THRESHOLD,
        message_batch_size=MESSAGE_BATCH_SIZE,
        score_transform=_score_transform,
        progress_every=10,
    )


if __name__ == "__main__":
    main()
