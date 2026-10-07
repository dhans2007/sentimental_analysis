import os
import sys

# Ensure the app directory is in the path so we can import app
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from app import predict_sentiment
except ImportError as e:
    print(f"Error importing predict_sentiment: {e}")
    sys.exit(1)

test_file = os.path.join(base_dir, "model", "test")

print("--- Testing Sentiment Model ---\n")
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line in ["Positive", "Negative", "Neutral"]:
            print(f"--- Expected Category: {line} ---")
            continue
        
        # Strip quotes if present
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
            
        try:
            label, confidence = predict_sentiment(line)
            print(f"Text: {line}")
            print(f"Sentiment: {label} (Confidence: {confidence:.4f})\n")
        except Exception as e:
            print(f"Failed to analyze: {line}. Error: {e}")
