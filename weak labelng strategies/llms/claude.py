import os
import json
from dotenv import load_dotenv

import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import anthropic
from datetime import datetime

# Current date and time
# now = datetime.now()
# print(now.strftime("%Y-%m-%d %H:%M:%S"))

# ---- Config ----
MODEL = "claude-sonnet-4-20250514"
OUT_CSV = "labeled_messages_claude_100.csv"

MESSAGES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/notebooks/messages_103.csv"
NARRATIVES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/data/Narratives.csv"

load_dotenv()  # loads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---- Helpers ----
def add_context(df):
    df = df.copy()
    df["context"] = df.apply(
        lambda r: f"{r.date_utc} {r.channel_username}",
        axis=1
    )
    return df


def build_candidate_list(narr_df):
    items = []
    for r in narr_df.itertuples():
        items.append({
            "narrative_id": str(r.narrative_id),
            "narrative": str(r.narrative),
            "sub_narrative_id": str(r.sub_narrative_id),
            "sub_narrative": str(r.sub_narrative),
        })
    return items


def _system_instructions_uk() -> str:
    return (
        "Ти уважний класифікатор. Поверни ЛИШЕ валідний JSON. Без зайвого тексту. "
        "НЕ вигадуй. НЕ використовуй markdown. "
        "НЕ перекладай і не перефразовуй текст наративів. "
        "У відповіді повертай ТІЛЬКИ narrative_id та sub_narrative_id (без тексту наративів)."
    )


def _task_instructions_uk() -> str:
    return (
        "Завдання: Визнач, чи повідомлення ПРЯМО й ЯВНО просуває будь-який наратив зі списку.\n"
        "Правила:\n"
        "- Спочатку виріши, чи є хоча б один відповідний наратив.\n"
        "- Відповідай «так» лише якщо твердження у повідомленні прямо підтримує наратив.\n"
        "- Якщо це загальні новини або нейтральний опис подій — НЕ став жодних наративів.\n"
        "- Використовуй ТІЛЬКИ narrative_id та sub_narrative_id зі списку.\n"
        "- Якщо жоден наратив не підходить — поверни null для narrative_id і sub_narrative_id та confidence 0.\n"
        "- Додай confidence від 0 до 1.\n"
        "- Вивід має бути валідним JSON.\n"
    )


TOOLS = [{
    "name": "label_message",
    "description": (
        "Вибери найкращий наратив/піднаратив для повідомлення або ніякий. "
        "Поверни лише narrative_id, sub_narrative_id, confidence і reason. "
        "Не повертай тексти наративів."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative_id": {"type": ["string", "null"]},
            "sub_narrative_id": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}
        },
        "required": ["narrative_id", "sub_narrative_id", "confidence", "reason"],
        "additionalProperties": False
    }
}]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5), reraise=True)
def classify_message(message_text, context_text, candidates_json):
    prompt = (
        f"{_task_instructions_uk()}\n"
        "Повідомлення:\n"
        f"{message_text}\n\n"
        "Контекст (канал та дата):\n"
        f"{context_text}\n\n"
        "JSON із наративами:\n"
        f"{candidates_json}\n"
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=_system_instructions_uk(),
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,
        tool_choice={
            "type": "tool",
            "name": "label_message",
            "disable_parallel_tool_use": True
        }
    )

    # Витягнути JSON із tool_use
    tool_block = next(
        b for b in resp.content
        if b.type == "tool_use" and b.name == "label_message"
    )
    return tool_block.input


def main():
    messages = pd.read_csv(MESSAGES_CSV)
    # messages = messages.sample(10, random_state=42)
    narratives = pd.read_csv(NARRATIVES_CSV)

    messages = add_context(messages)
    candidates = build_candidate_list(narratives)
    candidates_json = json.dumps(candidates, ensure_ascii=False)

    outputs = []
    for row in messages.itertuples():
        result = classify_message(row.text, row.context, candidates_json)
        outputs.append({
            "message_id": row.message_id,
            "narrative_id": result["narrative_id"],
            "sub_narrative_id": result["sub_narrative_id"],
            "confidence": result["confidence"],
            "reason": result["reason"]
        })

    out = pd.DataFrame(outputs)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
