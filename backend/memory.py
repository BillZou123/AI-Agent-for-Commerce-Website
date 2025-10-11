from collections import deque

# In-memory session store: { session_id: deque([...]) }
HISTORY = {}
MAX_TURNS = 50

def get_history(session_id):
    """Return a bounded deque for a session."""
    if session_id not in HISTORY:
        HISTORY[session_id] = deque(maxlen=MAX_TURNS)
    return HISTORY[session_id]