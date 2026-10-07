"""
NLP and Sentiment Analysis Utility Module
Provides aspect-based extraction, token-level explainability, emotion classification,
polarity scoring, model comparison, and social stream generation.
"""

import os
import re
import datetime
import random
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load fitted classical model artifacts as baseline/fallback
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
ensemble_model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))

# ---------------------------------------------------------------------------
# Deep Learning Transformer (DistilBERT loaded from model.safetensors)
# ---------------------------------------------------------------------------
transformer_model = None
transformer_tokenizer = None
USE_TRANSFORMER = False

try:
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertConfig
    from safetensors.torch import load_file

    potential_paths = [
        os.path.join(os.path.dirname(BASE_DIR), "model.safetensors"),
        os.path.join(BASE_DIR, "model", "model.safetensors"),
        os.path.join(BASE_DIR, "model.safetensors"),
        r"c:\Users\Dhanesh vc\Downloads\sentiment_analysis_project\model.safetensors"
    ]

    safetensors_path = None
    for p in potential_paths:
        if os.path.exists(p):
            safetensors_path = p
            break

    if safetensors_path:
        print(f"Loading DistilBERT weights from: {safetensors_path}")
        t_weights = load_file(safetensors_path)
        t_config = DistilBertConfig(
            vocab_size=30522, max_position_embeddings=512, sinusoidal_pos_embds=False,
            n_layers=6, n_heads=12, dim=768, hidden_dim=3072, num_labels=3
        )
        t_model = DistilBertForSequenceClassification(t_config)
        t_model.load_state_dict(t_weights)
        t_model.eval()

        t_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        transformer_model = t_model
        transformer_tokenizer = t_tokenizer
        USE_TRANSFORMER = True
        print("DistilBERT Transformer loaded successfully as primary inference engine!")
except Exception as e:
    print(f"Transformer initialization fallback to classical ensemble: {e}")
    transformer_model = None
    transformer_tokenizer = None
    USE_TRANSFORMER = False

# Curated sentiment word lists for word-level explainability
POSITIVE_LEXICON = {
    "amazing": 0.9, "awesome": 0.9, "excellent": 0.9, "perfect": 0.95, "love": 0.85,
    "loved": 0.85, "loving": 0.85, "fantastic": 0.9, "wonderful": 0.85, "best": 0.9,
    "great": 0.75, "good": 0.6, "snappy": 0.7, "fast": 0.65, "smooth": 0.7,
    "smoother": 0.75, "flawless": 0.95, "flawlessly": 0.95, "delightful": 0.85,
    "impressive": 0.8, "impressed": 0.8, "helpful": 0.7, "superb": 0.9,
    "brilliant": 0.85, "intuitive": 0.75, "outstanding": 0.9, "pleased": 0.7,
    "happy": 0.75, "thrilled": 0.9, "stellar": 0.85, "solid": 0.65, "reliable": 0.75,
    "clean": 0.6, "exceptional": 0.9, "indispensable": 0.85, "recommend": 0.75,
    "worth": 0.65, "delicious": 0.8, "crisp": 0.65, "easy": 0.6, "top": 0.7,
    "valuable": 0.7, "satisfied": 0.75, "satisfying": 0.75, "elegant": 0.8
}

NEGATIVE_LEXICON = {
    "terrible": 0.9, "horrible": 0.9, "worst": 0.95, "awful": 0.9, "bad": 0.7,
    "hate": 0.85, "hated": 0.85, "broken": 0.85, "crash": 0.8, "crashes": 0.85,
    "crashing": 0.85, "freeze": 0.75, "freezes": 0.8, "freezing": 0.8, "slow": 0.7,
    "lag": 0.7, "laggy": 0.75, "lags": 0.7, "glitch": 0.75, "glitches": 0.8,
    "bug": 0.7, "bugs": 0.75, "buggy": 0.8, "useless": 0.85, "disappointed": 0.85,
    "disappointing": 0.85, "ruined": 0.9, "waste": 0.85, "scam": 0.95, "garbage": 0.9,
    "trash": 0.9, "sucks": 0.85, "rude": 0.8, "ignored": 0.75, "poor": 0.75,
    "unacceptable": 0.85, "unusable": 0.9, "frustrating": 0.8, "frustrated": 0.8,
    "annoying": 0.7, "overpriced": 0.75, "dreadful": 0.9, "pathetic": 0.9,
    "abysmal": 0.95, "horrendous": 0.95, "fail": 0.8, "failed": 0.8, "failing": 0.8
}

# Aspect categories & domain keywords
ASPECT_DEFINITIONS = {
    "UI & Design": [
        "ui", "ux", "design", "layout", "interface", "look", "looks", "theme",
        "color", "colors", "dark mode", "visual", "buttons", "button", "screen",
        "aesthetic", "typography", "navigation"
    ],
    "Performance & Speed": [
        "speed", "fast", "slow", "snappy", "quick", "lag", "lags", "laggy",
        "freeze", "freezes", "crash", "crashes", "crashing", "performance",
        "battery", "memory", "ram", "cpu", "smooth", "stable", "stability"
    ],
    "Customer Support": [
        "support", "service", "agent", "agents", "help", "ticket", "tickets",
        "reply", "replies", "refund", "chat", "staff", "team", "care",
        "representative", "communication"
    ],
    "Pricing & Value": [
        "price", "pricing", "cost", "expensive", "cheap", "worth", "value",
        "deal", "subscription", "pay", "money", "free", "discount", "fee", "fees"
    ],
    "Features & Reliability": [
        "feature", "features", "update", "updates", "tool", "tools", "bug",
        "bugs", "glitch", "broken", "works", "working", "quality", "build",
        "reliable", "reliability", "durability"
    ]
}

# Emotion rules
EMOTION_KEYWORDS = {
    "Joy & Excitement": ["love", "loved", "loving", "amazing", "awesome", "great", "fantastic", "happy", "thrilled", "joy", "excited", "blessed", "wonderful", "celebrate"],
    "Frustration & Anger": ["hate", "hated", "angry", "furious", "annoying", "annoyed", "frustrated", "frustrating", "sucks", "awful", "scam", "waste", "garbage", "trash"],
    "Disappointment": ["disappointed", "disappointing", "regret", "sad", "sadly", "poor", "unfortunate", "ruined", "broken", "failed", "unhappy"],
    "Surprise & Awe": ["wow", "surprised", "shocked", "unexpected", "exceeded", "unbelievable", "stunned", "blown away", "remarkable", "game changer"],
    "Neutral & Informative": ["scheduled", "notice", "update", "release", "version", "specifications", "dimensions", "minutes", "hours", "guidelines", "report", "policy"]
}


def predict_sentiment_full(text):
    """
    Runs primary inference (DistilBERT Transformer if available, otherwise Classical Ensemble)
    and returns:
    - label: 'positive', 'neutral', 'negative'
    - confidence: float (0.0 to 1.0)
    - probabilities: dict with all 3 class probabilities
    - polarity_intensity: float from -1.0 (strongly negative) to +1.0 (strongly positive)
    """
    text_clean = (text or "").strip()
    if not text_clean:
        return "neutral", 0.33, {"positive": 0.33, "neutral": 0.34, "negative": 0.33}, 0.0

    if USE_TRANSFORMER and transformer_model is not None and transformer_tokenizer is not None:
        try:
            import torch
            inputs = transformer_tokenizer(text_clean, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = transformer_model(**inputs).logits.squeeze()
                probs = torch.softmax(logits, dim=-1).tolist()

            neg_p = probs[0]
            pos_p = probs[1]
            diff = abs(pos_p - neg_p)

            if diff < 0.18 and max(pos_p, neg_p) < 0.65:
                label = "neutral"
                neu_p = max(0.5, 1.0 - diff * 2.5)
                pos_p = (1.0 - neu_p) * 0.5 + (pos_p - neg_p) * 0.25
                neg_p = (1.0 - neu_p) - pos_p
                confidence = neu_p
            elif pos_p > neg_p:
                label = "positive"
                neu_p = max(0.02, 1.0 - pos_p - neg_p)
                confidence = pos_p
            else:
                label = "negative"
                neu_p = max(0.02, 1.0 - pos_p - neg_p)
                confidence = neg_p

            prob_dict = {
                "positive": round(float(pos_p), 4),
                "neutral": round(float(neu_p), 4),
                "negative": round(float(neg_p), 4)
            }
            intensity = round(float(pos_p - neg_p), 4)
            return label, round(float(confidence), 4), prob_dict, intensity
        except Exception as e:
            print(f"Transformer inference error, falling back to ensemble: {e}")

    # Fallback to classical Ensemble
    vec = vectorizer.transform([text_clean])
    label = ensemble_model.predict(vec)[0]
    probs = ensemble_model.predict_proba(vec)[0]
    classes = ensemble_model.classes_

    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
    confidence = round(float(max(probs)), 4)

    pos_p = prob_dict.get("positive", 0.0)
    neg_p = prob_dict.get("negative", 0.0)
    neu_p = prob_dict.get("neutral", 0.0)

    intensity = round((pos_p - neg_p) * (1.0 - (neu_p * 0.5)), 4)
    return label, confidence, prob_dict, intensity


def analyze_aspects(text):
    """
    Extracts relevant product/service aspects mentioned in text and scores each aspect.
    """
    text_lower = text.lower()
    aspect_results = []

    for aspect_name, keywords in ASPECT_DEFINITIONS.items():
        matched = []
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                matched.append(kw)

        if matched:
            # Extract local sentence context around the aspect keywords
            sentences = re.split(r"[.!?]+", text)
            relevant_sentences = []
            for s in sentences:
                s_strip = s.strip()
                if any(re.search(r"\b" + re.escape(kw) + r"\b", s_strip.lower()) for kw in matched):
                    relevant_sentences.append(s_strip)

            aspect_text = " ".join(relevant_sentences) if relevant_sentences else text
            label, conf, probs, intensity = predict_sentiment_full(aspect_text)

            aspect_results.append({
                "aspect": aspect_name,
                "keywords": matched,
                "sentiment": label,
                "confidence": conf,
                "intensity": intensity
            })

    return aspect_results


def explain_tokens(text):
    """
    Performs word-level polarity scoring to explain why the prediction was made.
    Returns a list of token objects: [{'text': str, 'sentiment': 'pos'|'neg'|'neu', 'score': float}]
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    explained = []

    negated = False
    for t in tokens:
        w_lower = t.lower()
        if w_lower in [".", ",", "!", "?", ";", ":"]:
            negated = False
            explained.append({"word": t, "type": "punct", "weight": 0.0})
        elif w_lower in ["not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely", "without"]:
            negated = True
            explained.append({"word": t, "type": "negator", "weight": -0.4})
        elif w_lower in POSITIVE_LEXICON:
            score = POSITIVE_LEXICON[w_lower]
            if negated:
                explained.append({"word": t, "type": "neg", "weight": -score})
            else:
                explained.append({"word": t, "type": "pos", "weight": score})
        elif w_lower in NEGATIVE_LEXICON:
            score = NEGATIVE_LEXICON[w_lower]
            if negated:
                explained.append({"word": t, "type": "pos", "weight": score * 0.6})
            else:
                explained.append({"word": t, "type": "neg", "weight": -score})
        else:
            explained.append({"word": t, "type": "neu", "weight": 0.0})

    return explained


def detect_emotion(text, primary_sentiment):
    """
    Detects the dominant emotional tone of the text.
    """
    text_lower = text.lower()
    scores = {}

    for emotion, kws in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in kws if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        scores[emotion] = count

    # Boost based on primary sentiment
    if primary_sentiment == "positive":
        scores["Joy & Excitement"] += 1
    elif primary_sentiment == "negative":
        scores["Frustration & Anger"] += 1
    else:
        scores["Neutral & Informative"] += 2

    best_emotion = max(scores, key=scores.get)
    return best_emotion


def compare_all_models(text):
    """
    Runs DistilBERT transformer alongside all 4 underlying sub-models and the ensemble
    to allow comprehensive side-by-side benchmarking.
    """
    text_clean = (text or "").strip()
    if not text_clean:
        return {}

    results = {}

    # 1. DistilBERT Transformer (Deep Neural Network)
    if USE_TRANSFORMER and transformer_model is not None and transformer_tokenizer is not None:
        try:
            import torch
            inputs = transformer_tokenizer(text_clean, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = transformer_model(**inputs).logits.squeeze()
                probs = torch.softmax(logits, dim=-1).tolist()

            neg_p = probs[0]
            pos_p = probs[1]
            diff = abs(pos_p - neg_p)

            if diff < 0.18 and max(pos_p, neg_p) < 0.65:
                t_label = "neutral"
                neu_p = max(0.5, 1.0 - diff * 2.5)
                pos_p = (1.0 - neu_p) * 0.5 + (pos_p - neg_p) * 0.25
                neg_p = (1.0 - neu_p) - pos_p
                t_conf = neu_p
            elif pos_p > neg_p:
                t_label = "positive"
                neu_p = max(0.02, 1.0 - pos_p - neg_p)
                t_conf = pos_p
            else:
                t_label = "negative"
                neu_p = max(0.02, 1.0 - pos_p - neg_p)
                t_conf = neg_p

            results["DistilBERT Transformer (Deep Neural Network)"] = {
                "label": t_label,
                "confidence": round(float(t_conf), 4),
                "probs": {
                    "positive": round(float(pos_p), 4),
                    "neutral": round(float(neu_p), 4),
                    "negative": round(float(neg_p), 4)
                }
            }
        except Exception as e:
            results["DistilBERT Transformer (Deep Neural Network)"] = {
                "label": "error", "confidence": 0.0, "error": str(e)
            }

    # 2. Ensemble & Classical Models
    vec = vectorizer.transform([text_clean])

    ens_label = str(ensemble_model.predict(vec)[0])
    ens_probs = ensemble_model.predict_proba(vec)[0]
    results["Soft-Voting Ensemble (TF-IDF Baseline)"] = {
        "label": ens_label,
        "confidence": round(float(max(ens_probs)), 4),
        "probs": {str(cls): round(float(p), 4) for cls, p in zip(ensemble_model.classes_, ens_probs)}
    }

    model_labels = {
        "svc": "Linear Support Vector Classifier (LinearSVC)",
        "lr": "L2-Regularized Logistic Regression",
        "cnb": "Complement Naive Bayes (CNB)",
        "sgd": "Modified Huber SGD Classifier"
    }

    for name, est in ensemble_model.named_estimators_.items():
        friendly_name = model_labels.get(name, name)
        try:
            raw_pred = est.predict(vec)[0]
            if isinstance(raw_pred, (int, np.integer)) or str(raw_pred).isdigit():
                p_label = str(ensemble_model.classes_[int(raw_pred)])
            else:
                p_label = str(raw_pred)

            if hasattr(est, "predict_proba"):
                probs = est.predict_proba(vec)[0]
                conf = round(float(max(probs)), 4)
                p_dict = {str(ensemble_model.classes_[i]): round(float(probs[i]), 4) for i in range(len(ensemble_model.classes_))}
            else:
                conf = 0.90
                p_dict = {p_label: 0.90}

            results[friendly_name] = {
                "label": p_label,
                "confidence": float(conf),
                "probs": p_dict
            }
        except Exception as e:
            results[friendly_name] = {"label": "error", "confidence": 0.0, "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# AI Auto-Responder & Action Item Extraction Engine (Powered by Gemini + Fallback)
# ---------------------------------------------------------------------------
def generate_ai_reply_with_gemini(text, label, confidence, emotion, aspects_list, custom_instruction=None):
    """
    Attempts to generate context-aware, personalized response drafts and team actions
    using Google Gemini GenAI SDK. Returns None if key is missing or call fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or api_key.startswith("your_gemini_api_key"):
        return None

    try:
        from google import genai
        from google.genai import types
        import json

        client = genai.Client(api_key=api_key.strip())

        system_instruction = (
            "You are an elite, world-class AI Customer Experience & Support specialist. "
            "Your task is to analyze customer feedback, tickets, or reviews and produce:\n"
            "1. Urgency assessment ('critical', 'high', 'medium', or 'low') with a clean emoji title (e.g. 'CRITICAL 🚨', 'HIGH ⚠️', 'MEDIUM ℹ️', 'LOW 🌟').\n"
            "2. 2-4 concrete, highly relevant team action items with appropriate department tags (e.g. '⚙️ [Engineering] ...', '💳 [Billing] ...', '📦 [Logistics] ...', '🎧 [Customer Success] ...').\n"
            "3. 3 highly personalized, context-aware customer response drafts tailored specifically to their message:\n"
            "   - 'empathetic': Warm, deeply understanding, caring, sincere, de-escalating.\n"
            "   - 'professional': Courteous, structured, objective, business-standard.\n"
            "   - 'solution': Action-oriented, direct, offering clear next steps, workarounds, or resolutions.\n"
            "All responses MUST directly address the exact details, products, emotions, and specific issues mentioned in the customer text."
        )

        prompt_data = {
            "customer_message": text,
            "detected_sentiment": label,
            "detected_emotion": emotion,
            "detected_aspects": aspects_list,
            "custom_instruction": custom_instruction or "None provided"
        }

        prompt = (
            f"Analyze this customer feedback and generate urgency, action items, and 3 distinct reply drafts.\n\n"
            f"Customer Data:\n{json.dumps(prompt_data, indent=2)}\n\n"
            f"Return ONLY a valid JSON object matching this schema:\n"
            f"{{\n"
            f'  "urgency": "CRITICAL 🚨",\n'
            f'  "urgency_code": "critical",\n'
            f'  "action_items": ["⚙️ [Engineering] ...", "🎧 [Support] ..."],\n'
            f'  "replies": {{\n'
            f'    "empathetic": "Dear Customer,\\n\\n...",\n'
            f'    "professional": "Hello,\\n\\n...",\n'
            f'    "solution": "Hi there,\\n\\n..."\n'
            f'  }}\n'
            f"}}"
        )

        # Try gemini-2.5-flash or fallback model names
        model_names = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
        response = None
        last_err = None

        for m in model_names:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.7,
                    )
                )
                if response and response.text:
                    break
            except Exception as me:
                last_err = me
                continue

        if not response or not response.text:
            if last_err:
                print(f"Gemini API model attempts failed: {last_err}")
            return None

        clean_text = response.text.strip()
        # Clean markdown wrappers if any
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        parsed = json.loads(clean_text.strip())
        if "replies" in parsed and "action_items" in parsed:
            return parsed
        return None
    except Exception as e:
        print(f"Gemini generation fallback notice: {e}")
        return None


def generate_smart_reply_and_actions(text, custom_instruction=None):
    """
    Analyzes customer review/feedback and generates:
    - Sentiment & emotion metadata
    - Urgency level (Critical, High, Medium, Low)
    - Key extracted action items for team execution
    - 3 Tailored Response Drafts (Empathetic, Professional, Solution)
    Uses Google Gemini AI if configured, otherwise falls back to intelligent rule engine.
    """
    label, confidence, probs, intensity = predict_sentiment_full(text)
    aspects = analyze_aspects(text)
    emotion = detect_emotion(text, label)
    aspects_list = [a["aspect"] for a in aspects]
    text_lower = text.lower()

    # 1. Attempt AI Generation with Gemini
    ai_result = generate_ai_reply_with_gemini(
        text=text,
        label=label,
        confidence=confidence,
        emotion=emotion,
        aspects_list=aspects_list,
        custom_instruction=custom_instruction
    )

    if ai_result:
        return {
            "text": text,
            "sentiment": label,
            "confidence": confidence,
            "emotion": emotion,
            "aspects": aspects_list,
            "urgency": ai_result.get("urgency", "MEDIUM ℹ️"),
            "urgency_code": ai_result.get("urgency_code", "medium"),
            "action_items": ai_result.get("action_items", []),
            "replies": ai_result.get("replies", {}),
            "ai_powered": True,
            "provider": "Google Gemini AI"
        }

    # 2. Local Fallback Engine
    if any(k in text_lower for k in ["crash", "crashing", "bricked", "lost data", "lawyer", "scam", "unauthorized", "stolen", "chargeback"]):
        urgency = "CRITICAL 🚨"
        urgency_code = "critical"
    elif label == "negative" or any(k in text_lower for k in ["broken", "refund", "horrible", "damaged", "fail", "terrible", "worst"]):
        urgency = "HIGH ⚠️"
        urgency_code = "high"
    elif label == "neutral" or any(k in text_lower for k in ["delay", "slow", "pricing", "confusing", "question", "how to"]):
        urgency = "MEDIUM ℹ️"
        urgency_code = "medium"
    else:
        urgency = "LOW 🌟"
        urgency_code = "low"

    action_items = []
    if any(k in text_lower for k in ["crash", "bug", "freeze", "error", "glitch", "broken"]):
        action_items.append("⚙️ [Engineering] File high-priority bug report and reproduce issue from user logs.")
    if any(k in text_lower for k in ["refund", "billing", "charged", "cost", "price", "overpriced"]):
        action_items.append("💳 [Billing] Review customer account for potential credit or refund processing.")
    if any(k in text_lower for k in ["shipping", "delivery", "arrived", "damaged", "package"]):
        action_items.append("📦 [Logistics] Check fulfillment status and dispatch replacement unit if needed.")
    if any(k in text_lower for k in ["ui", "design", "layout", "confusing", "hard to use"]):
        action_items.append("🎨 [Product/UX] Log user interface feedback into the next design sprint backlog.")
    if any(k in text_lower for k in ["support", "agent", "service", "reply", "ignored"]):
        action_items.append("🎧 [Customer Success] Escalate ticket to senior support supervisor for follow-up.")
    if label == "positive":
        action_items.append("⭐ [Marketing] Request customer review permission or NPS testimonial feature.")
    if not action_items:
        action_items.append("📋 [General] Archive feedback into the customer experience repository.")

    if label == "negative":
        draft_empathetic = (
            f"Dear Customer,\n\n"
            f"Thank you for bringing this to our attention. We are genuinely sorry to hear about your experience. "
            f"This is certainly not the standard of quality we strive for. Our team has already logged this issue with our engineering and support leads. "
            f"Please reply directly to this message or contact us at support@sentiment-scope.com with your account details so we can make this right immediately.\n\n"
            f"Warm regards,\nCustomer Care Team"
        )
        draft_professional = (
            f"Hello,\n\n"
            f"Thank you for contacting us. We have received your feedback regarding our service. "
            f"We take reports of this nature very seriously and have initiated an internal review to investigate the root cause. "
            f"A member of our support team will reach out to you within 24 hours with a resolution.\n\n"
            f"Sincerely,\nSupport Operations"
        )
        draft_solution = (
            f"Hi there,\n\n"
            f"We deeply appreciate your candid feedback and apologize for the inconvenience caused. "
            f"We would love the opportunity to fix this for you right away. Could you please share a screenshot or error details with us at priority@sentiment-scope.com? "
            f"We are on standby to assist you.\n\n"
            f"Best,\nCustomer Success Team"
        )
    elif label == "positive":
        draft_empathetic = (
            f"Hi there!\n\n"
            f"Thank you so much for the wonderful feedback! Messages like yours make our entire team's day. "
            f"We're thrilled to know that you're enjoying the experience. If you ever have ideas or suggestions for new features, we'd love to hear them.\n\n"
            f"Warmly,\nThe Product Team"
        )
        draft_professional = (
            f"Dear Customer,\n\n"
            f"Thank you for taking the time to share your positive experience with us. "
            f"We are delighted to hear that our service met your expectations. We look forward to continuing to provide you with the highest quality service.\n\n"
            f"Sincerely,\nCustomer Relations Team"
        )
        draft_solution = (
            f"Hello!\n\n"
            f"We are so grateful for your support and thrilled that you love using our platform! "
            f"Would you be open to sharing your review on our public community page? Either way, thank you for being a valued user!\n\n"
            f"Best wishes,\nCommunity Team"
        )
    else:
        draft_empathetic = (
            f"Hello,\n\n"
            f"Thank you for reaching out to us. We have noted your inquiry and feedback. "
            f"If there is any additional information or specific assistance you need regarding this update, please let us know and we'll be happy to help.\n\n"
            f"Best regards,\nCustomer Support Team"
        )
        draft_professional = (
            f"Dear Customer,\n\n"
            f"Thank you for your message. We confirm receipt of your details regarding the scheduled timeline. "
            f"For detailed specifications and updates, please refer to our documentation portal or reply if you need further clarification.\n\n"
            f"Sincerely,\nOperations Team"
        )
        draft_solution = (
            f"Hi,\n\n"
            f"Thanks for keeping us informed! Our team is monitoring all upcoming rollouts closely. "
            f"Feel free to check our status portal for live release schedules.\n\n"
            f"Best,\nSupport Team"
        )

    return {
        "text": text,
        "sentiment": label,
        "confidence": confidence,
        "emotion": emotion,
        "aspects": aspects_list,
        "urgency": urgency,
        "urgency_code": urgency_code,
        "action_items": action_items,
        "replies": {
            "empathetic": draft_empathetic,
            "professional": draft_professional,
            "solution": draft_solution
        },
        "ai_powered": False,
        "provider": "Built-in Rule Engine (Add GEMINI_API_KEY in .env for Gemini AI)"
    }



# ---------------------------------------------------------------------------
# Web URL Content Scraper & Full-Article Sentiment Intelligence
# ---------------------------------------------------------------------------
from html.parser import HTMLParser
import urllib.request
import urllib.parse


class WebpageTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.in_title = False
        self.in_ignored_tag = 0
        self.paragraphs = []
        self.current_chunk = []
        self.ignored_tags = {"script", "style", "nav", "footer", "header", "noscript", "svg", "button", "input"}

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower == "title":
            self.in_title = True
        elif tag_lower in self.ignored_tags:
            self.in_ignored_tag += 1
        elif tag_lower in {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "article", "section"}:
            self.flush_chunk()

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower == "title":
            self.in_title = False
        elif tag_lower in self.ignored_tags:
            self.in_ignored_tag = max(0, self.in_ignored_tag - 1)
        elif tag_lower in {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "article", "section"}:
            self.flush_chunk()

    def handle_data(self, data):
        if self.in_ignored_tag > 0:
            return
        if self.in_title:
            self.title += data
        else:
            text = data.strip()
            if text:
                self.current_chunk.append(text)

    def flush_chunk(self):
        if self.current_chunk:
            full_text = " ".join(self.current_chunk).strip()
            if len(full_text) > 30 and not full_text.startswith("{") and not full_text.startswith("Copyright"):
                self.paragraphs.append(full_text)
            self.current_chunk = []


DEMO_URL_PRESETS = {
    "preset://iphone-review": {
        "title": "Comprehensive Review: The Next Generation Flagship Experience",
        "domain": "techinsights.io",
        "paragraphs": [
            "The latest hardware generation is an absolute masterclass in industrial engineering and aesthetic design.",
            "Display color reproduction is vivid and the 120Hz adaptive refresh rate feels silky smooth throughout all daily tasks.",
            "The battery longevity easily exceeded all our benchmark targets, comfortably delivering two full days of intensive productivity.",
            "However, the charging speed remains painfully slow compared to competitor devices in the same luxury price bracket.",
            "Furthermore, customer support response times were rather disappointing when resolving early shipment delays.",
            "Overall, this device represents an outstanding upgrade for anyone seeking top-tier performance and reliable construction."
        ]
    },
    "preset://cloud-incident": {
        "title": "Post-Mortem: Incident Report on Global API Latency and Outages",
        "domain": "cloudstatus.net",
        "paragraphs": [
            "On September 14, an unexpected routing table misconfiguration triggered elevated latency across our European data centers.",
            "The incident caused annoying connection dropouts and severe timeouts for approximately 45 minutes.",
            "Our automated failover systems took longer than expected to reroute traffic, frustrating several production customers.",
            "Our engineering team intervened swiftly, verified the root cause, and successfully deployed a permanent configuration patch.",
            "All services have resumed normal operations with zero data loss or database corruption reported.",
            "We sincerely apologize for the disruption and have implemented additional redundant sanity checks."
        ]
    },
    "preset://saas-launch": {
        "title": "Product Announcement: Announcing Analytics Engine v3.0",
        "domain": "saasplatform.com",
        "paragraphs": [
            "We are thrilled to announce the official general availability of our next-generation Analytics Engine v3.0!",
            "Dashboard load times have been drastically reduced by 75%, making complex data exploration instant and delightful.",
            "The new dark mode user interface is clean, modern, and beautifully crafted for high-density monitoring.",
            "Pricing for new enterprise tiers starts at $99 monthly with unlimited team seats and 24/7 dedicated support.",
            "All existing customers have been automatically upgraded with zero downtime or manual migration required.",
            "We would like to express our deepest gratitude to our beta community for their invaluable suggestions and feedback."
        ]
    }
}


def scrape_and_analyze_url(target_url):
    """
    Scrapes a webpage or loads a preset, breaks content into sentences,
    and calculates comprehensive sentiment intelligence metrics.
    """
    target_url = (target_url or "").strip()
    if not target_url:
        raise ValueError("No URL provided.")

    if target_url in DEMO_URL_PRESETS:
        preset = DEMO_URL_PRESETS[target_url]
        title = preset["title"]
        domain = preset["domain"]
        paragraphs = preset["paragraphs"]
    else:
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            target_url = "https://" + target_url

        parsed = urllib.parse.urlparse(target_url)
        domain = parsed.netloc or "webpage"
        title = f"Content from {domain}"
        paragraphs = []
        fetch_error = None

        # Attempt 1: Direct HTTP fetch with modern browser headers
        req = urllib.request.Request(
            target_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                html_bytes = response.read()
                html_text = html_bytes.decode("utf-8", errors="ignore")
                parser = WebpageTextExtractor()
                parser.feed(html_text)
                parser.flush_chunk()
                if parser.paragraphs:
                    paragraphs = parser.paragraphs[:15]
                    if parser.title.strip():
                        title = parser.title.strip()
        except Exception as e:
            fetch_error = e

        # Attempt 2: If direct fetch was blocked (e.g. Medium/Cloudflare 403) or no text was found, use reader fallback
        if not paragraphs:
            try:
                proxy_url = f"https://r.jina.ai/{target_url}"
                proxy_req = urllib.request.Request(
                    proxy_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(proxy_req, timeout=12) as response:
                    content_text = response.read().decode("utf-8", errors="ignore")

                lines = content_text.split("\n")
                extracted_pars = []
                for line in lines:
                    l = line.strip()
                    if l.startswith("Title:") and title.startswith("Content from"):
                        extracted_title = l.replace("Title:", "").strip()
                        if extracted_title:
                            title = extracted_title
                    elif len(l) > 35 and not l.startswith("URL Source:") and not l.startswith("Markdown Content:") and not l.startswith("![") and not l.startswith("[!["):
                        clean_l = re.sub(r"\[.*?\]\(.*?\)", "", l).strip()
                        clean_l = re.sub(r"^#+\s*", "", clean_l).strip()
                        if len(clean_l) > 30:
                            extracted_pars.append(clean_l)

                if extracted_pars:
                    paragraphs = extracted_pars[:15]
            except Exception as proxy_err:
                if fetch_error:
                    raise RuntimeError(f"Could not fetch webpage from {target_url}: {fetch_error}")
                else:
                    raise RuntimeError(f"Could not fetch webpage from {target_url}: {proxy_err}")

    if not paragraphs:
        raise RuntimeError("No readable article text could be extracted from this URL.")

    # Break paragraphs into individual sentences and analyze each
    sentences_data = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    total_intensity = 0.0
    aspect_counts = {"UI & Design": 0, "Performance & Speed": 0, "Customer Support": 0, "Pricing & Value": 0, "Features & Reliability": 0}

    for p in paragraphs:
        raw_sents = re.split(r"(?<=[.!?])\s+", p)
        for s in raw_sents:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue

            label, conf, probs, intensity = predict_sentiment_full(s_clean)
            aspects = analyze_aspects(s_clean)
            counts[label] += 1
            total_intensity += intensity

            for a in aspects:
                asp_name = a["aspect"]
                if asp_name in aspect_counts:
                    aspect_counts[asp_name] += 1

            sentences_data.append({
                "text": s_clean,
                "sentiment": label,
                "confidence": conf,
                "intensity": intensity,
                "aspects": [a["aspect"] for a in aspects]
            })

    total_sentences = len(sentences_data) or 1
    avg_intensity = round(total_intensity / total_sentences, 4)

    # Calculate overall article verdict
    pos_pct = round((counts["positive"] / total_sentences) * 100, 1)
    neu_pct = round((counts["neutral"] / total_sentences) * 100, 1)
    neg_pct = round((counts["negative"] / total_sentences) * 100, 1)

    # Net Sentiment Score: Pos% - Neg% (-100 to +100)
    net_sentiment_score = round(pos_pct - neg_pct, 1)

    if net_sentiment_score > 15:
        overall_sentiment = "positive"
    elif net_sentiment_score < -15:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    # Top positive & negative quotes
    sorted_pos = sorted([s for s in sentences_data if s["sentiment"] == "positive"], key=lambda x: x["confidence"], reverse=True)
    sorted_neg = sorted([s for s in sentences_data if s["sentiment"] == "negative"], key=lambda x: x["confidence"], reverse=True)

    top_positive_quotes = [s["text"] for s in sorted_pos[:3]]
    top_negative_quotes = [s["text"] for s in sorted_neg[:3]]

    # Word count & estimated read time
    total_words = sum(len(s["text"].split()) for s in sentences_data)
    read_time_min = max(1, round(total_words / 200))

    # Executive TL;DR summary
    summary_bullets = [
        f"Analyzed {total_sentences} sentences ({total_words} words) from '{domain}'.",
        f"Overall Tone: {overall_sentiment.upper()} with a Net Sentiment Score of {net_sentiment_score:+.1f}.",
        f"Sentiment Distribution: {pos_pct}% Positive, {neu_pct}% Neutral, {neg_pct}% Negative.",
    ]
    if top_positive_quotes:
        summary_bullets.append(f"Primary Praise: \"{top_positive_quotes[0]}\"")
    if top_negative_quotes:
        summary_bullets.append(f"Main Concern: \"{top_negative_quotes[0]}\"")

    return {
        "url": target_url,
        "title": title,
        "domain": domain,
        "total_words": total_words,
        "read_time_min": read_time_min,
        "total_sentences": total_sentences,
        "overall_sentiment": overall_sentiment,
        "net_sentiment_score": net_sentiment_score,
        "avg_intensity": avg_intensity,
        "sentiment_breakdown": {
            "positive": counts["positive"],
            "neutral": counts["neutral"],
            "negative": counts["negative"],
            "positive_pct": pos_pct,
            "neutral_pct": neu_pct,
            "negative_pct": neg_pct
        },
        "aspect_distribution": aspect_counts,
        "top_positive_quotes": top_positive_quotes,
        "top_negative_quotes": top_negative_quotes,
        "summary_bullets": summary_bullets,
        "sentences": sentences_data
    }
