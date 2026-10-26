# AutoMLens

Audit AutoML on clinical tabular data: tuning comparison, FLAML vs AutoGluon, leakage canary, bias, SHAP, model card, FastAPI and Next.js demo.
Educational project; not a medical device.

## Run (Windows PowerShell, Python 3.11 and Node 18+ installed)
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
. .\.venv\Scripts\Activate.ps1
pip install autogluon.tabular          # optional, large

# dataset: put your Kaggle token at $env:USERPROFILE\.kaggle\kaggle.json first
kaggle datasets download -d sulianova/cardiovascular-disease-dataset -p data --unzip

pytest -q
python -m automlens.cli --budget 900   # use --budget 120 --skip-autogluon for a quick trial
uvicorn api.main:app --reload --port 8000
```
## Web UI (second terminal)
```powershell
.\setup-web.ps1
cd web
npx skills add Leonxlnx/taste-skill
npm run dev                            # http://localhost:3000
```
## Deploy
```powershell
# API: push the repo to GitHub, then create a Render Web Service (Docker) or a Hugging Face Docker Space.
# Frontend:
cd web
npm i -g vercel
vercel login
vercel --prod
vercel env add NEXT_PUBLIC_API_URL production   # paste your deployed API URL
vercel --prod
```
