import os
import json
import logging
import matplotlib.pyplot as plt

def save_report(report, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    json_report_path = os.path.join(output_dir, "compatibility_report.json")
    try:
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Report saved to {json_report_path}")
    except Exception as e:
        logging.error(f"Failed to save report: {e}", exc_info=True)

def save_sequence_length_histogram(sequence_lengths, output_dir):
    if not sequence_lengths:
        logging.warning("No sequence lengths to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50, color='blue', edgecolor='black')
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    histogram_path = os.path.join(output_dir, 'sequence_length_distribution.png')
    plt.savefig(histogram_path)
    plt.close()
    logging.info(f"Sequence length histogram saved to {histogram_path}")