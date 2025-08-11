# NatCat Clustering Assessment

This project is an end-to-end pipeline for clustering article titles extracted from the GDELT API into five natural catastrophe groups:

- **Earthquake**
- **Floods**
- **Volcano**
- **Tornado**
- **Wildfire**

The workflow covers data extraction, cleaning, feature engineering, clustering, and evaluation, with a focus on scalable and reproducible data science practices.

## Project Structure

```
├── abbrevations.json
├── Contractions_lowercase.json
├── requirements.txt
├── data/
│   ├── [Cleaned and processed CSV files at each stage]
│   ├── Nat Cat Events - Raw data.csv
│   ├── scaled_features_for_models.npz
│   ├── model_predictions/
│   │   ├── Agglomerative_predictions.csv
│   │   ├── GMM_predictions.csv
│   │   └── kmeans_predictions.csv
│   └── model_weights/
│       ├── agglomerative_weights.joblib/.pickle
│       ├── gmm_weights.joblib/.pickle
│       └── kmeans_weights.joblib/.pickle
├── notebooks/
│   ├── 1. Exploratory Data Analysis.ipynb
│   ├── 2. Data Cleaning and NatCat detection.ipynb
│   └── 3. Feature Engineering and Clustering.ipynb
├── scripts/
│   ├── Clean_data.py
│   └── helpers/
│       ├── data_cleaning.py
│       ├── lemmatise.py
│       └── natcat_detection.py
├── reports/
│   ├── Evaluation.docx
│   └── NatCat Clustering Presentation.pptx
```

### Folder Descriptions

- **notebooks/**: Contains all the analysis and step-by-step workflow in separate Jupyter notebooks:
  - **1. Exploratory Data Analysis.ipynb**: Initial data exploration and visualization.
  - **2. Data Cleaning and NatCat detection.ipynb**: Data preprocessing, cleaning, and natural catastrophe detection logic.
  - **3. Feature Engineering and Clustering.ipynb**: Feature extraction, scaling, clustering, and evaluation.

- **scripts/**: Python scripts for efficient data processing. These scripts replicate the cleaning and transformation steps from the notebooks, but are optimized for speed using multiprocessing. Helper scripts are modularized for reusability. Running these scripts outputs intermediate and final CSV files to the `data/` folder.

- **data/**: Stores all data files used and generated in the project, including:
  - Raw input data from GDELT
  - Cleaned and processed CSVs at each stage
  - Scaled feature files for modeling
  - Model weights for each clustering algorithm (Agglomerative, GMM, KMeans)
  - Model prediction outputs for each clustering method

- **reports/**: Contains the final evaluation report and presentation summarizing the project findings and results.

- **requirements.txt**: List of all Python dependencies. Install these before running any scripts or notebooks.

## Setup Instructions

1. **Clone the repository** (if not already):
	```
	git clone <repo-url>
	cd <folder>
	```

2. **Install dependencies**:
	```
	pip install -r requirements.txt
	```

3. **Run the workflow**:
	- Explore the notebooks in order for a step-by-step walkthrough.
	- For faster data processing, use the scripts in the `scripts/` folder. Outputs will be saved in the `data/` directory.

## Notes

- The project is modular and reproducible. All intermediate and final outputs are saved for transparency and reusability.
- For any custom abbreviations or contractions handling, refer to the provided JSON files.

---

For any questions or clarifications, please refer to the notebooks.
