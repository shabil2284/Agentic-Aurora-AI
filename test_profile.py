import cProfile
import pstats
import io
import time
import tempfile
import sys

# Mock Streamlit session state
class DummySessionState(dict):
    def __getattr__(self, k):
        return self.get(k)
    def __setattr__(self, k, v):
        self[k] = v

import streamlit as st
st.session_state = DummySessionState()

def run_load():
    bytes_data = b"dummy pdf content" * 1024 * 1024 # 1MB
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(bytes_data)
        st.session_state.current_file_path = tmp.name
        st.session_state.raw_file_bytes = bytes_data

start = time.time()
import app # This simulates the full script execution Streamlit does
print(f"app.py import taking {time.time() - start:.2f}s")

pr = cProfile.Profile()
pr.enable()

run_load()

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())
