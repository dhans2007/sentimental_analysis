import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sentiment_status_dir = os.path.join(base_dir, "sentiment_status")
sys.path.insert(0, sentiment_status_dir)

from sentiment_status.app import app
