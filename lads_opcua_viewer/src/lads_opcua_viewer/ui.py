"""Small helpers for Streamlit containers."""

import time


# MARK: empty
def empty(container):
    container.empty()
    time.sleep(0.02)
    return container
