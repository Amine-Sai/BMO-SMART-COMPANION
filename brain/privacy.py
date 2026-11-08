from collections.abc import Mapping
from typing import Any

import numpy as np

MAX_NUMERIC_ARRAY_LEN = 32


def sanitize_payload(obj: Any, max_array_len: int = MAX_NUMERIC_ARRAY_LEN) -> Any:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<redacted:bytes:len={len(obj)}>"
    if isinstance(obj, np.ndarray):
        return f"<redacted:ndarray:shape={obj.shape}:dtype={obj.dtype}>"
    if isinstance(obj, Mapping):
        return {str(k): sanitize_payload(v, max_array_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_array_len and all(
            isinstance(x, (int, float, np.number)) and not isinstance(x, bool) for x in obj
        ):
            return f"<redacted:numeric_array:len={len(obj)}>"
        return [sanitize_payload(x, max_array_len) for x in obj]
    return obj
