import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from embedding_utils import run_similarity_pipeline

# ---- Config ----
MODEL = "BAAI/bge-m3"
# MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT_CSV = "labeled_messages_hf_embeddings_bge_m3_100.csv"

MESSAGES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/notebooks/messages_103.csv"
NARRATIVES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/data/Narratives.csv"

NARRATIVE_TEMPLATE = "{sub_narrative}"
MESSAGE_PREFIX_CHARS = None

EMBED_BATCH_SIZE = 64
MESSAGE_BATCH_SIZE = 64
THRESHOLD = 0.75  # applied after score_transform


def _score_transform(cos_sim: float) -> float:
    return (cos_sim + 1.0) / 2.0


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    messages = pd.read_csv(MESSAGES_CSV)
    # messages = messages.sample(10, random_state=42)
    narratives = pd.read_csv(NARRATIVES_CSV)

    device = resolve_device()
    model = SentenceTransformer(MODEL, device=device)

    def embed_texts(texts: list[str]) -> list[list[float]]:
        return model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).tolist()

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
