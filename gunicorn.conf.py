"""Small-instance defaults; Render supplies PORT at runtime."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
# Each worker loads its own model and knowledge graph. Start with one on Free.
workers = int(os.environ.get('WEB_CONCURRENCY', '1'))
threads = 1
timeout = 120
accesslog = '-'
errorlog = '-'
