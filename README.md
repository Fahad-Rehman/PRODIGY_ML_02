# Customer Segmentation using K-Means Clustering

## Project Overview
This project implements customer segmentation using K-Means clustering on mall customer data. The pipeline preprocesses data, finds the optimal number of clusters using the elbow method, trains a K-Means model, and visualizes the results.

## Project Structure
```text
customer-segmentation/
├── data/
│ ├── raw/
│ │ └── Mall_Customers.csv
│ └── processed/
│ └── processed_data.csv
├── models/
│ └── kmeans_model.pkl
├── results/
│ ├── plots/
│ │ ├── elbow_plot.png
│ │ ├── silhouette_plot.png
│ │ └── cluster_visualization.png
│ ├── cluster_centers.csv
│ └── clustered_data.csv
├── src/
│ ├── data_preprocessing.py
│ ├── best_k.py
│ ├── train_model.py
│ └── visualize.py
├── main.py
├── requirements.txt
└── README.md
```


## Features
- Data preprocessing and standardization
- Automatic optimal cluster selection using elbow method and silhouette scores
- K-Means clustering implementation
- Visualization of clusters and evaluation metrics
- Model persistence and results export

## Installation
1. Clone this repository
    ```bash
    git clone <repository-url>
    cd Task2
    ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the complete pipeline:
```bash
python main.py
```

## Individual Components
+ Data Preprocessing: `src/data_preprocessing.py`
+ Find Optimal K: `src/best_k.py`
+ Train Model: `src/train_model.py`
+ Visualize Results: `src/visualize.py`

## Data
The dataset contains customer information including:

+ Age
+ Annual Income (k$)
+ Spending Score (1-100)

## Outputs
+ Models: Trained K-Means model saved as .pkl file

+ Results:
    + Cluster assignments (`clustered_data.csv`)
    + Cluster centers (`cluster_centers.csv`)
    + Visualization plots in `results/plots/`
    + Elbow method and silhouette score plots

## Dependencies
+ pandas
+ scikit-learn
+ matplotlib
+ seaborn
+ joblib
+ numpy

## License
This project is for educational purposes as part of the Prodigy Infotech internship program.


