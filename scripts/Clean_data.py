
"""
Clean_data.py
-------------
Script for cleaning and processing Natural Catastrophe event data using parallel processing.
Produces three CSVs in the data/ folder:
    1. Events_cleaned_basic.csv
    2. Events_with_natcat_flag.csv
    3. Events_fully_cleaned_lemmatised.csv
"""

import os
import warnings
import logging
import pandas as pd
from pandarallel import pandarallel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(processName)s | %(message)s'
)

# URLs for external resources
CONTRACTIONS_URL = 'https://raw.githubusercontent.com/ShravanTV/Natural_Catastrophe_Events/refs/heads/main/Contractions_lowercase.json'
ACRONYMS_URL = 'https://raw.githubusercontent.com/ShravanTV/Natural_Catastrophe_Events/refs/heads/main/abbrevations.json'

# Initialize pandarallel for parallel processing
pandarallel.initialize(progress_bar=True, nb_workers=3, verbose=1)

# ========== Pandarallel Wrapper Functions ==========
def detect_natcat_event(text):
    """
    Wrapper for NatCat event detection (pandarallel friendly).
    Imports and instantiates detector inside the worker process.
    """
    from helpers.natcat_detection import NatCatEventDetector
    detector = NatCatEventDetector()
    return detector.is_nat_cat_event(text)

def lemmatize_wrapper(text):
    """
    Wrapper for lemmatization (pandarallel friendly).
    Imports inside the worker process.
    """
    from helpers.lemmatise import lemmatise_text
    return lemmatise_text(text)

# ========== Data Processing Functions ==========
def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean event titles using DataCleaner (parallelized).
    """
    from helpers.data_cleaning import DataCleaner
    cleaner = DataCleaner(acronyms_url=ACRONYMS_URL, contractions_url=CONTRACTIONS_URL)
    df['cleaned_title'] = df['title'].parallel_apply(cleaner.clean_text)
    return df

def save_csv(df: pd.DataFrame, filename: str, folder: str = "data"):
    """
    Save DataFrame to CSV in the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, filename)
    df.to_csv(out_path, index=False)
    logging.info(f"Saved: {out_path}")

# ========== Main Pipeline ==========
def main():
    INPUT_CSV = os.path.join("data", "Test_titles.csv")
    logging.info(f"Reading input file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Drop duplicates and missing titles
    df = df.drop_duplicates(subset=['title'])
    df = df.dropna(subset=['title'])
    df['title'] = df['title'].astype(str).fillna('')

    # 1. Basic cleaning
    logging.info("Cleaning event titles...")
    df = clean_titles(df)
    save_csv(df, "1.Events_cleaned_basic.csv")

    # Remove rows with missing/duplicate cleaned titles
    df = df.dropna(subset=['cleaned_title'])
    df = df.drop_duplicates(subset=['cleaned_title'])

    # 2. NatCat event detection
    logging.info("Detecting NatCat events...")
    df['is_natcat'] = df['cleaned_title'].parallel_apply(detect_natcat_event)
    save_csv(df, "2.Events_with_natcat_flag.csv")
    logging.info(f"NatCat detection complete. Found {df['is_natcat'].sum()} NatCat events.")

    # 3. Filter only NatCat events
    natcat_df = df[df['is_natcat'] == True].copy()
    if natcat_df.empty:
        logging.warning("No NatCat events found after filtering.")
    else:
        # Lemmatize titles
        logging.info("Lemmatizing titles...")
        natcat_df['lemmatised_title'] = natcat_df['cleaned_title'].parallel_apply(lemmatize_wrapper)
        save_csv(natcat_df, "3.Events_fully_cleaned_lemmatised.csv")
        logging.info("Lemmatization complete.")

if __name__ == "__main__":
    main()

    