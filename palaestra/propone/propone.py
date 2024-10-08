import json
import logging
from typing import List, Dict, Any
from jsonpath_rw import parse as jsonpath_parse
from .utils import clean_text, transform_value

def process_data(input_file: str, output_file: str, mappings: List[Dict[str, str]]) -> None:
    logging.info(f"Starting data processing from {input_file} to {output_file}")
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_number, line in enumerate(f_in, start=1):
            try:
                record = json.loads(line)
                processed_record = {}
                for mapping in mappings:
                    output_field = mapping['output_field']
                    json_path = mapping['json_path']
                    json_path_expr = jsonpath_parse(json_path)
                    matches = json_path_expr.find(record)
                    if matches:
                        values = [match.value for match in matches]
                        transformed_value = transform_value(values)
                        processed_record[output_field] = transformed_value
                    else:
                        processed_record[output_field] = ''
                        logging.warning(f"No match for JSON path '{json_path}' at line {line_number}.")
                json.dump(processed_record, f_out, ensure_ascii=False)
                f_out.write('\n')
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error at line {line_number}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error at line {line_number}: {e}")
    logging.info(f"Data processing completed. Output saved to {output_file}.")

def interactive_mapping(sample_record: Dict[str, Any]) -> List[Dict[str, str]]:
    print("Entering interactive mapping mode.")
    print("Sample record:")
    print(json.dumps(sample_record, indent=4, ensure_ascii=False))
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

def propone(config: Dict[str, Any]) -> None:
    input_file = config['input_file']
    output_file = config['output_file']
    mappings = config.get('mappings')

    if not mappings:
        logging.info("No mappings provided. Starting interactive mapping.")
        with open(input_file, 'r', encoding='utf-8') as f:
            sample_record = json.loads(f.readline())
        mappings = interactive_mapping(sample_record)

    process_data(input_file, output_file, mappings)