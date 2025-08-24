import warnings

ORANGE = "\033[38;5;215m"  # Código 208 = naranja aproximado en 256 colores
RESET = "\033[0m"

def warn(msg: str):
    import warnings
    original_format = warnings.formatwarning
    try:
        warnings.formatwarning = lambda m, *a, **k: f"{ORANGE}⚠️⚠️⚠️ {m}{RESET} ⚠️⚠️⚠️\n"
        warnings.warn(msg, UserWarning)
    finally:
        warnings.formatwarning = original_format
