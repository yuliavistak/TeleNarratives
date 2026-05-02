import re
import pandas as pd

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove @mentions and hashtags symbols (keep the words)
    text = re.sub(r"[@#]", " ", text)

    text = re.sub(r"[^a-zа-яіїєґё\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def cleaning_and_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for cleaning dataset of messages
    """
    df = df.drop_duplicates()

    df = df[
        df['text'].notna() &
        (df['text'].astype(str).str.len() >= 20)
    ]
    df = df[df['text'].str.contains(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", regex=True)] # deleting messages with emojis/punctuation only - 27

    df['date_utc'] = pd.to_datetime(df['date_utc'], errors='coerce', utc=True)
    df = df.dropna(subset=['date_utc'])

    df['text'] = df['text'].apply(clean_text)

    df['is_forwarded'][(df['is_forwarded']==True)&(df['fwd_from_channel_id'].isna())&(df['fwd_from_message_id'].isna())] = False

    return df

def stratified_sample_by_week(df, output_path, n_samples, random_state=42):
    df = df.copy()

    iso = df['date_utc'].dt.isocalendar()
    df['_week_of_year'] = iso.week.astype(int)

    week_distribution = df['_week_of_year'].value_counts(normalize=True)

    samples = []
    for week, frac in week_distribution.items():
        n_week = int(round(n_samples * frac))
        subset = df[df['_week_of_year'] == week]
        if n_week > 0:
            samples.append(
                subset.sample(n=min(len(subset), n_week), random_state=random_state)
            )

    sampled = pd.concat(samples, ignore_index=True).drop(columns=['_week_of_year'])
    sampled.to_csv(output_path, index=False)

    remaining = df.merge(
        sampled[['channel_id', 'message_id']],
        on=['channel_id', 'message_id'],
        how='left',
        indicator=True
    )
    remaining = remaining[remaining['_merge'] == 'left_only'].drop(columns=['_merge', '_week_of_year'])


    return remaining