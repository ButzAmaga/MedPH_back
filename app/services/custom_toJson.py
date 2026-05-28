from datetime import date, datetime
import json
import math
import cupy as np
import cudf as pd


class SafeJsonEncoder(json.JSONEncoder):
    """
    Extends the default JSON encoder to handle types that commonly appear
    in cudf snapshot dicts but are not JSON-serialisable by default:
 
    - datetime / date          → ISO-8601 string
    - cupy integers           → int
    - cupy floats             → float  (NaN / Inf → None)
    - cupy booleans           → bool
    - cupy arrays / pd.Series → list
    - pd.NA / pd.NaT           → None
    - any remaining unknown    → str(obj) as a safe fallback
    """
 
    def default(self, obj):
        # ── datetime family ────────────────────────────────────────────────
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if obj is pd.NaT:
            return None
 
        # ── cupy scalars ──────────────────────────────────────────────────
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(obj, np.bool_):
            return bool(obj)
 
        # ── array-like ─────────────────────────────────────────────────────
        if isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
            return obj.tolist()
 
        # ── cudf NA ──────────────────────────────────────────────────────
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
 
        # ── safe fallback ──────────────────────────────────────────────────
        return str(obj)