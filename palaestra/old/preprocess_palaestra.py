import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    logging.info("Starting preprocessing")
    # ... (rest of the main function)
    logging.info("Preprocessing completed")

if __name__ == '__main__':
    main()