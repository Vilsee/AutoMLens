"""Step 1: load, clean and split the Cardiovascular Disease dataset (Kaggle, ~70k rows)."""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL = 'cardio'
FEATURES = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active', 'age_years', 'bmi']


def load(path='data/cardio_train.csv'):
    df = pd.read_csv(path, sep=';').drop(columns='id')
    raw = len(df)
    df['age_years'] = (df.age / 365.25).round(1)
    df = df.drop(columns='age')
    # remove physiologically impossible values
    df = df[df.ap_hi.between(70, 250) & df.ap_lo.between(40, 150) & (df.ap_hi > df.ap_lo)]
    df = df[df.height.between(120, 220) & df.weight.between(30, 200)]
    df['bmi'] = df.weight / (df.height / 100) ** 2
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[FEATURES + [LABEL]]
    df.attrs['raw_rows'] = raw
    return df


def split(df, test_size=0.2):
    tr, te = train_test_split(df, test_size=test_size, stratify=df[LABEL], random_state=SEED)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
