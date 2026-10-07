# Social Media Sentiment Analysis Application

Cloud-hosted web app that classifies social media comments/tweets/reviews as
**positive**, **negative**, or **neutral**, built for deployment on
**Cloud Foundry**.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, vanilla JS |
| Backend | Python, Flask |
| ML Model | scikit-learn (TF-IDF + Logistic Regression) |
| Database | MongoDB (MongoDB Atlas free tier / CF Mongo service) |
| Deployment | Cloud Foundry (`manifest.yml`, `Procfile`, `python_buildpack`) |

## Current Status

**Working:**
- Flask app with 5 routes: `/` (analyze form), `/analyze` (POST, runs the ML
  model), `/history` (past analyses), `/api/stats` (sentiment counts for the
  breakdown bars), `/healthz` (readiness check)
- Sentiment classifier trained locally: TF-IDF vectorizer + Logistic
  Regression, ~170-row labeled dataset (`data/training_data.csv`), **71%
  test accuracy**
- MongoDB integration via `pymongo`, with a `MONGO_URI` env var — reads/writes
  analyzed comments and their sentiment/confidence/timestamp
- App degrades gracefully with no DB connection (still analyzes text, just
  doesn't persist) — verified locally without a running MongoDB instance
- Frontend dashboard: textarea + "Analyze Sentiment" button, live sentiment
  badge with confidence %, sentiment breakdown bars, history page
- Cloud Foundry deployment files ready: `manifest.yml`, `Procfile`,
  `runtime.txt`, `requirements.txt`

**Not done yet / next steps:**
- Not yet deployed to an actual Cloud Foundry sandbox (Red Hat OpenShift Dev
  Sandbox / SAP BTP CF trial) — deployment files are written and tested
  structurally, but a live `cf push` hasn't been run
- MongoDB Atlas free-tier cluster not yet provisioned — app currently tested
  against "no DB" fallback path only
- Classifier accuracy (71%) is limited by the small training set — plan to
  expand the dataset (or fine-tune on a public Twitter sentiment dataset) to
  improve it
- No live social media API integration yet (e.g., Twitter/X API) — currently
  takes manually pasted text, not a live feed

## Project Structure

```
sentiment_status/
├── app.py                  # Flask app + routes
├── model/
│   ├── train_model.py      # Trains & saves the TF-IDF + LogisticRegression model
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── sentiment_model.pkl # Trained classifier
├── data/
│   └── training_data.csv   # Labeled training sentences
├── templates/               # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   └── history.html
├── static/
│   ├── css/style.css
│   └── js/script.js
├── requirements.txt
├── Procfile                 # gunicorn start command
├── manifest.yml             # Cloud Foundry app manifest
├── runtime.txt
└── .env.example
```

## Running Locally

```bash
pip install -r requirements.txt
python model/train_model.py     # only needed if you change training_data.csv
python app.py                   # runs on http://localhost:5000
```

Optionally set `MONGO_URI` (see `.env.example`) to persist analyses; without
it the app still runs, it just won't save history.

## Deploying to Cloud Foundry

```bash
cf login -a <api-endpoint>
cf create-service mongodb-atlas free sentiment-mongodb   # or bind an existing Mongo Atlas instance via a user-provided service
cf push
```

`manifest.yml` binds the `sentiment-mongodb` service and starts the app with
gunicorn. Cloud Foundry injects the Mongo connection details via
`VCAP_SERVICES` when the service is bound — for the current version, that
gets passed through as `MONGO_URI`.

## API

| Route | Method | Description |
|---|---|---|
| `/` | GET | Analyze form |
| `/analyze` | POST | `{"text": "..."}` → `{"sentiment", "confidence", "text", "timestamp"}` |
| `/history` | GET | Last 50 analyzed comments |
| `/api/stats` | GET | `{"positive": n, "neutral": n, "negative": n}` |
| `/healthz` | GET | `{"status": "ok", "db_connected": bool}` |
