import os
import io
import csv
import json
import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from dotenv import load_dotenv

from nlp_utils import (
    predict_sentiment_full,
    analyze_aspects,
    explain_tokens,
    detect_emotion,
    generate_smart_reply_and_actions,
    scrape_and_analyze_url
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# ---------------------------------------------------------------------------
# Backward-compatible function for test runners and external scripts
# ---------------------------------------------------------------------------
def predict_sentiment(text):
    label, confidence, _, _ = predict_sentiment_full(text)
    return label, confidence


# ---------------------------------------------------------------------------
# MongoDB connection with graceful local in-memory fallback
# ---------------------------------------------------------------------------
collection = None
feedback_collection = None

try:
    from pymongo import MongoClient

    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.admin.command("ping")
    db = client["sentiment_db"]
    collection = db["analyses"]
    feedback_collection = db["feedback"]
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"MongoDB not available, running with in-memory persistence: {e}")
    collection = None
    feedback_collection = None

local_analyses = []
local_feedback = []


# ---------------------------------------------------------------------------
# HTML Page Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", active_page="analyze")


@app.route("/batch")
def batch():
    return render_template("batch.html", active_page="batch")


@app.route("/assistant")
def assistant():
    return render_template("assistant.html", active_page="assistant")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/url-analyzer")
def url_analyzer():
    return render_template("url_analyzer.html", active_page="url-analyzer")


@app.route("/history")
def history():
    records = []
    if collection is not None:
        try:
            records = list(collection.find().sort("timestamp", -1).limit(100))
            for r in records:
                r["_id"] = str(r["_id"])
        except Exception as e:
            print(f"Could not fetch history: {e}")
    else:
        records = sorted(local_analyses, key=lambda x: x.get("timestamp", ""), reverse=True)[:100]
    return render_template("history.html", records=records, active_page="history")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html", active_page="how-it-works")


@app.route("/theme.css")
def theme_css():
    return send_from_directory(os.path.join(BASE_DIR, "static", "css"), "theme.css", mimetype="text/css")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    label, confidence, probs, intensity = predict_sentiment_full(text)
    aspects = analyze_aspects(text)
    tokens = explain_tokens(text)
    emotion = detect_emotion(text, label)

    record = {
        "text": text,
        "sentiment": label,
        "confidence": confidence,
        "probabilities": probs,
        "intensity": intensity,
        "aspects": [a["aspect"] for a in aspects],
        "aspect_details": aspects,
        "tokens": tokens,
        "emotion": emotion,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # Save to database or local storage
    if collection is not None:
        try:
            collection.insert_one(dict(record))
        except Exception as e:
            print(f"Could not save to MongoDB: {e}")
    else:
        local_analyses.append(record)

    return jsonify(record)


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """
    Handles CSV or JSON batch file uploads or raw text line arrays.
    """
    items = []

    if "file" in request.files:
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename.lower()

        if filename.endswith(".csv") or filename.endswith(".txt"):
            stream = io.StringIO(uploaded_file.stream.read().decode("utf-8", errors="ignore"))
            reader = csv.reader(stream)
            rows = list(reader)
            if rows:
                start_idx = 1 if any(h in rows[0][0].lower() for h in ["text", "comment", "review", "tweet", "feedback"]) else 0
                for r in rows[start_idx:]:
                    if r and r[0].strip():
                        items.append(r[0].strip())
        elif filename.endswith(".json"):
            content = json.loads(uploaded_file.stream.read().decode("utf-8", errors="ignore"))
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and item.strip():
                        items.append(item.strip())
                    elif isinstance(item, dict) and "text" in item:
                        items.append(str(item["text"]).strip())
    else:
        data = request.get_json(silent=True) or {}
        items = data.get("texts", [])

    if not items:
        return jsonify({"error": "No valid text items found in upload."}), 400

    items = items[:500]

    results = []
    summary_counts = {"positive": 0, "neutral": 0, "negative": 0}

    for text in items:
        label, conf, probs, intensity = predict_sentiment_full(text)
        aspects = analyze_aspects(text)
        emotion = detect_emotion(text, label)
        summary_counts[label] += 1

        rec = {
            "text": text,
            "sentiment": label,
            "confidence": conf,
            "intensity": intensity,
            "emotion": emotion,
            "aspects": [a["aspect"] for a in aspects],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        results.append(rec)

        if collection is not None:
            try:
                collection.insert_one(dict(rec))
            except Exception:
                pass
        else:
            local_analyses.append(rec)

    return jsonify({
        "total": len(results),
        "summary": summary_counts,
        "results": results
    })


@app.route("/api/assistant", methods=["POST"])
def api_assistant():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    custom_instruction = (data.get("custom_instruction") or "").strip() or None
    if not text:
        return jsonify({"error": "No review or message provided"}), 400

    result = generate_smart_reply_and_actions(text, custom_instruction=custom_instruction)
    return jsonify(result)



@app.route("/api/scrape-url", methods=["POST"])
def api_scrape_url():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        results = scrape_and_analyze_url(url)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """
    Returns comprehensive analytics for the interactive dashboard.
    """
    records = []
    if collection is not None:
        try:
            records = list(collection.find().sort("timestamp", -1).limit(500))
        except Exception as e:
            print(f"Could not fetch stats records: {e}")
    else:
        records = local_analyses

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    emotions = {}
    aspect_counts = {"UI & Design": 0, "Performance & Speed": 0, "Customer Support": 0, "Pricing & Value": 0, "Features & Reliability": 0}
    timeline = {}

    for r in records:
        sent = r.get("sentiment", "neutral")
        if sent in counts:
            counts[sent] += 1

        emo = r.get("emotion") or detect_emotion(r.get("text", ""), sent)
        emotions[emo] = emotions.get(emo, 0) + 1

        for asp in r.get("aspects", []):
            if asp in aspect_counts:
                aspect_counts[asp] += 1

        ts = r.get("timestamp", "")
        if ts:
            date_key = ts[:10]
            if date_key not in timeline:
                timeline[date_key] = {"positive": 0, "neutral": 0, "negative": 0}
            timeline[date_key][sent] = timeline[date_key].get(sent, 0) + 1

    timeline_sorted = [{"date": k, **v} for k, v in sorted(timeline.items())[-7:]]

    return jsonify({
        "counts": counts,
        "total": sum(counts.values()),
        "emotions": emotions,
        "aspects": aspect_counts,
        "timeline": timeline_sorted
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    predicted = data.get("predicted", "")
    corrected = data.get("corrected", "")
    feedback_type = data.get("type", "correct")

    feedback_doc = {
        "text": text,
        "predicted": predicted,
        "corrected": corrected,
        "feedback_type": feedback_type,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    if feedback_collection is not None:
        try:
            feedback_collection.insert_one(feedback_doc)
        except Exception as e:
            print(f"Error saving feedback: {e}")
    else:
        local_feedback.append(feedback_doc)

    return jsonify({"status": "recorded", "message": "Thank you for the feedback!"})


@app.route("/api/history/delete", methods=["POST"])
def api_history_delete():
    global local_analyses
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")

    if action == "clear_all":
        if collection is not None:
            try:
                collection.delete_many({})
            except Exception as e:
                print(f"Error clearing mongo: {e}")
        local_analyses = []
        return jsonify({"status": "cleared"})

    return jsonify({"error": "Invalid action"}), 400


@app.route("/api/history/export", methods=["GET"])
def api_history_export():
    records = []
    if collection is not None:
        try:
            records = list(collection.find().sort("timestamp", -1))
        except Exception:
            records = local_analyses
    else:
        records = local_analyses

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Text", "Sentiment", "Confidence", "Emotion", "Aspects", "Timestamp"])

    for r in records:
        writer.writerow([
            r.get("text", ""),
            r.get("sentiment", ""),
            r.get("confidence", ""),
            r.get("emotion", ""),
            ", ".join(r.get("aspects", [])),
            r.get("timestamp", "")
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sentiment_analysis_export.csv"}
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "db_connected": collection is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
