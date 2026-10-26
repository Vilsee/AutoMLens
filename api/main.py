import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

ART = Path(__file__).resolve().parent.parent / 'artifacts'
FEATURES = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active', 'age_years', 'bmi']

app = FastAPI(title='AutoMLens API', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
model = joblib.load(ART / 'model.joblib')
try:
    import shap
    explainer = shap.TreeExplainer(model)
except Exception:  # SHAP is optional at serve time
    explainer = None


class Patient(BaseModel):
    gender: int = Field(ge=1, le=2, description='1 = woman, 2 = man')
    height: float = Field(ge=120, le=220, description='cm')
    weight: float = Field(ge=30, le=200, description='kg')
    ap_hi: int = Field(ge=70, le=250, description='systolic mmHg')
    ap_lo: int = Field(ge=40, le=150, description='diastolic mmHg')
    cholesterol: int = Field(ge=1, le=3)
    gluc: int = Field(ge=1, le=3)
    smoke: int = Field(ge=0, le=1)
    alco: int = Field(ge=0, le=1)
    active: int = Field(ge=0, le=1)
    age_years: float = Field(ge=18, le=100)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(p: Patient):
    row = pd.DataFrame([p.model_dump()])
    row['bmi'] = row.weight / (row.height / 100) ** 2
    row = row[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    out = {'risk': round(prob, 4), 'label': int(prob >= 0.5), 'contributions': []}
    if explainer is not None:
        sv = explainer.shap_values(row)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        pairs = sorted(zip(FEATURES, sv[0]), key=lambda x: -abs(x[1]))[:6]
        out['contributions'] = [{'feature': f, 'impact': round(float(v), 4)} for f, v in pairs]
    return out


@app.get('/leaderboard')
def leaderboard():
    return json.loads((ART / 'results.json').read_text(encoding='utf-8'))


@app.get('/model-card', response_class=PlainTextResponse)
def model_card():
    return (ART / 'MODEL_CARD.md').read_text(encoding='utf-8')
