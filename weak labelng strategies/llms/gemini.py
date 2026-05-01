import json
from dotenv import load_dotenv

import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from google import genai

from datetime import datetime

now = datetime.now()

# Format: 2026-02-20 12:23:52
print(now.strftime("%Y-%m-%d %H:%M:%S"))
# ---- Config ----
PART = 12000
MODEL = "gemini-2.5-flash"  # підтримує structured outputs
OUT_CSV = f"labeled_similarity_messages_gemini_sheet1.csv"

MESSAGES_CSV = f"/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/notebooks/labeling/strategies/Forwards to - Sheet1.csv"
NARRATIVES_CSV = "/Users/yuliavistak/Desktop/UCU/Навчання/4 курс/diploma/disinfo_graph/notebooks/labeling/strategies/labeling - sub-narratives.csv"

load_dotenv()  # читає GEMINI_API_KEY з .env
client = genai.Client()

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
        "Ти уважний український класифікатор. Поверни ЛИШЕ валідний JSON. Без зайвого тексту. "
        "НЕ вигадуй. НЕ використовуй markdown. "
        "НЕ перекладай і не перефразовуй текст наративів. "
        "У відповіді повертай ТІЛЬКИ narrative_id та sub_narrative_id (без тексту наративів)."
    )

def _task_instructions_uk() -> str:
    return (
        "Завдання: Визнач, чи повідомлення ПРЯМО або НЕПРЯМО ('натякаючи') просуває будь-який наратив зі списку.\n"
        "Правила:\n"
        "- Спочатку виріши, чи є хоча б один відповідний наратив.\n"
        "- Відповідай «так» лише якщо твердження у повідомленні прямо підтримує наратив.\n"
        "- Якщо це загальні новини або нейтральний опис подій — НЕ став жодних наративів.\n"
        "- Може бути таке, що повідомлення може містити кілька наративів. Обери той, що найбільше підходить, або найбільш чіткий.\n"
        "- Читай уважно. Дивись на повідомлення з української перспективи.\n"
        "- Використовуй ТІЛЬКИ narrative_id та sub_narrative_id зі списку.\n"
        "- Якщо жоден наратив не підходить — поверни null для narrative_id і sub_narrative_id та confidence 0.\n"
        "- Додай confidence від 0 до 1.\n"
        "- Вивід має бути валідним JSON.\n"
        """
        - Наприклад:\n
        1. "кличко заявив що україна стикається з браком солдатів і пропонує знизити вік мобілізації деталі за посиланням" - це повідомлення хоче нав'язує думку, що Кличко хоче знизити мобілізаційний вік, це страшить громадян і у них виникає погана думка про політика Кличка, тому тут є наратив "Дискредитація чи висміювання представників української влади"\n    
        2. "мужики красавчики понимая что ударные дроны вылетают из контейнера на прицепе фуры начали забрасывать его камнями рискуя своей жизнью конечно они понимали об опасности было бы хорошо если бы их нашли и наградили просто красавчики neoficialniybezsonov" - повідомлення стимулює 'захоплюватись' сміливістю росіян і хоче переконати, що перемога росіян однозначна, тому тут присутній наратив "Перемога України неможлива"\n
        """
    )

# JSON schema для structured outputs
SCHEMA = {
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


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def classify_message(message_text, context_text, candidates_json):
    prompt = (
        f"{_system_instructions_uk()}\n\n"
        f"{_task_instructions_uk()}\n"
        "Повідомлення:\n"
        f"{message_text}\n\n"
        "Контекст (канал та дата):\n"
        f"{context_text}\n\n"
        "JSON із наративами:\n"
        f"{candidates_json}\n"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": SCHEMA,
            "temperature": 0
        }
    )
    return json.loads(response.text)


def main():
    messages = pd.read_csv(MESSAGES_CSV)
    narratives = pd.read_csv(NARRATIVES_CSV)

    messages = add_context(messages)
    candidates = build_candidate_list(narratives)
    candidates_json = json.dumps(candidates, ensure_ascii=False)

    outputs = []
    idx = 0
    total = len(messages)
    for row in messages.itertuples():
        idx += 1
        result = classify_message(row.text, row.context, candidates_json)
        outputs.append({
            "channel_id": row.channel_id,
            "message_id": row.message_id,
            "narrative_id": result["narrative_id"],
            "sub_narrative_id": result["sub_narrative_id"],
            "confidence": result["confidence"],
            "reason": result["reason"]
        })
        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total}")
            out = pd.DataFrame(outputs)
            out.to_csv(OUT_CSV, index=False)
        if idx % 100 == 0 or idx == total:
            now = datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))

    out = pd.DataFrame(outputs)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
