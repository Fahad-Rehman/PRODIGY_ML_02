import os
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def train_model(best_k,
                data_path="data/processed/processed_data.csv",
                model_path="models/kmeans_model.pkl",
                result_path="results/"):

    # Create directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)
    print(f"Training KMeans with k={best_k}")

    # Train final model
    model = KMeans(n_clusters=best_k, random_state=42)
    labels = model.fit_predict(data)

    # Save trained model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save cluster labels to CSV
    labeled_data = data.copy()
    labeled_data["cluster"] = labels

    labeled_csv_path = os.path.join(result_path, "clustered_data.csv")
    labeled_data.to_csv(labeled_csv_path, index=False)
    print(f"Clustered data saved to: {labeled_csv_path}")

    # Save cluster centers
    centers_path = os.path.join(result_path, "cluster_centers.csv")
    pd.DataFrame(model.cluster_centers_).to_csv(centers_path, index=False)
    print(f"Cluster centers saved to: {centers_path}")

    #Plot clusters if data has 2 columns
    if data.shape[1] == 2:
        plt.figure(figsize=(6, 4))
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, s=30)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                    s=200, marker='X')
        plt.title(f"KMeans Clusters (k={best_k})")
        cluster_plot_path = os.path.join(result_path, "cluster_plot.png")
        plt.savefig(cluster_plot_path)
        plt.close()
        print(f"Cluster plot saved to: {cluster_plot_path}")

    print("\nTraining complete.\n")

    return model, labels, data.values
