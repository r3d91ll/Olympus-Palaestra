#!/usr/bin/env python3
# preprocess_palaestra.py

"""
Preprocessing Utility for Palaestra's Gymnasia

This script preprocesses complex JSON datasets with nested structures,
transforming them into a flat format suitable for model training.
It supports configuration via a YAML file and provides interactive
guidance if mappings are not specified.

Usage:
    python preprocess_palaestra.py --config preprocess_config.yaml
"""

import json
import yaml
import argparse
import logging
import os
import sys
from jsonpath_rw import parse as jsonpath_parse
from typing import Any, Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the preprocessing configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        sys.exit(1)

def save_output(output_data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save the preprocessed data to a file in JSON Lines format.

    Args:
        output_data (List[Dict[str, Any]]): List of preprocessed records.
        output_file (str): Path to the output file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            for record in output_data:
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')
        logging.info(f"Preprocessing complete. Output saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        sys.exit(1)

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
    import re
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

def main():
    parser = argparse.ArgumentParser(description='Preprocess JSON data for Palaestra\'s Gymnasia.')
    parser.add_argument('--config', type=str, help='Path to the preprocess_config.yaml file.')
    args = parser.parse_args()

    if not args.config:
        logging.error("Configuration file is required. Use --config to specify the path.")
        sys.exit(1)

    config = load_config(args.config)
    input_file = config.get('input_file')
    output_file = config.get('output_file')
    mappings = config.get('mappings', [])

    if not input_file or not output_file:
        logging.error("Input and output file paths must be specified in the configuration.")
        sys.exit(1)

    # If mappings are not provided, enter interactive mode
    if not mappings:
        logging.info("No mappings provided in configuration. Starting interactive mode.")
        with open(input_file, 'r', encoding='utf-8') as file:
            first_line = file.readline()
            try:
                record_sample = json.loads(first_line)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing first line of input file: {e}")
                sys.exit(1)
        mappings = interactive_mode(record_sample)

    output_data = parse_and_transform(input_file, mappings)
    save_output(output_data, output_file)

if __name__ == '__main__':
    main()