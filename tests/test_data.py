import pandas as pd
from automlens import data


def test_split_is_stratified(tmp_path):
    rows = [dict(id=i, age=18000 + i * 37, gender=1 + i % 2, height=150 + i % 60, weight=50 + i % 40,
                 ap_hi=110 + i % 50, ap_lo=70 + i % 20, cholesterol=1 + i % 3, gluc=1, smoke=0, alco=0,
                 active=1, cardio=i % 2) for i in range(400)]
    p = tmp_path / 'd.csv'
    pd.DataFrame(rows).to_csv(p, sep=';', index=False)
    df = data.load(str(p))
    assert len(df) == 400
    tr, te = data.split(df)
    assert abs(tr.cardio.mean() - te.cardio.mean()) < 0.02
