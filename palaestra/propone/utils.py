import re
from typing import Any, Optional

def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize text by removing extra whitespace.

    Args:
        text (Optional[str]): The text to clean.

    Returns:
        str: Cleaned text.
    """
    if text is None:
        return ''
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def transform_value(values: Any) -> str:
    """
    Transform the extracted values into a string suitable for the model.

    Args:
        values (Any): The extracted value(s) from the JSON path.

    Returns:
        str: Transformed string value.
    """
    if isinstance(values, list):
        # Join list elements into a single string
        text = ' '.join(map(str, values))
    else:
        text = str(values)
    return clean_text(text)