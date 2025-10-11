from collections import deque

# ---------- Session Memory ----------
# Stores chat history for each session in memory.
# HISTORY is a dictionary keyed by session_id.
# Each session keeps a bounded deque of turns (user/assistant messages).

HISTORY = {}
MAX_TURNS = 50  # cap the number of turns per session

def get_history(session_id):
    """
    Retrieve the chat history for a given session.
    If no history exists, create a new deque with MAX_TURNS capacity.
    """
    if session_id not in HISTORY:
        HISTORY[session_id] = deque(maxlen=MAX_TURNS)
    return HISTORY[session_id]