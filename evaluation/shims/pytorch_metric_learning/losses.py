def __getattr__(name):
    return lambda *a, **k: None
