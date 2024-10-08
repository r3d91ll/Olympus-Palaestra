import json
import logging
from typing import Any, Dict, List
from jsonpath_rw import parse as jsonpath_parse
from .utils import transform_value

def parse_and_transform(input_file: str, mappings: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Parse the input JSON file and apply the specified mappings to transform the data.

    Args:
        input_file (str): Path to the input JSON file.
        mappings (List[Dict[str, str]]): List of mappings from JSON paths to output fields.

    Returns:
        List[Dict[str, Any]]: List of transformed records.
    """
    output_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                transformed_record = {}
                for mapping in mappings:
                    output_field = mapping['output_field']
                    json_path = mapping['json_path']
                    json_path_expr = jsonpath_parse(json_path)
                    matches = json_path_expr.find(record)
                    if matches:
                        values = [match.value for match in matches]
                        transformed_value = transform_value(values)
                        transformed_record[output_field] = transformed_value
                    else:
                        transformed_record[output_field] = ''
                        logging.warning(f"No match for JSON path '{json_path}' at line {line_number}.")
                output_data.append(transformed_record)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error at line {line_number}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error at line {line_number}: {e}")
    return output_data

def preprocess_data(input_file: str, output_file: str, mappings: List[Dict[str, str]]) -> None:
    """
    Preprocess the input data and save the results.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output file.
        mappings (List[Dict[str, str]]): List of mappings from JSON paths to output fields.
    """
    output_data = parse_and_transform(input_file, mappings)
    save_output(output_data, output_file)

def save_output(output_data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save the preprocessed data to a file in JSON Lines format.

    Args:
        output_data (List[Dict[str, Any]]): List of preprocessed records.
        output_file (str): Path to the output file.
    """
    import os
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            for record in output_data:
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')
        logging.info(f"Preprocessing complete. Output saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        raise