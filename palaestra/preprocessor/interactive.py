import json
from typing import Any, Dict, List

def interactive_mode(record_sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Provide an interactive CLI to assist the user in specifying mappings.

    Args:
        record_sample (Dict[str, Any]): A sample record from the dataset.

    Returns:
        List[Dict[str, str]]: List of user-defined mappings.
    """
    print("Entering interactive mode.")
    print("Sample record:")
    print(json.dumps(record_sample, indent=4, ensure_ascii=False))
    mappings = []
    while True:
        output_field = input("\nEnter the output field name (or type 'done' to finish): ")
        if output_field.lower() == 'done':
            break
        json_path = input(f"Enter the JSON path for '{output_field}': ")
        mappings.append({
            'output_field': output_field,
            'json_path': json_path
        })
    return mappings