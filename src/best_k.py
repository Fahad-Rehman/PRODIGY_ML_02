import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_best_k(data_path="data/processed/processed_data.csv",
                result_path="results/plots"):
    
    os.makedirs(result_path, exist_ok=True)

    # Load the data
    data = pd.read_csv(data_path)

    inertia_values = []
    silhouette_scores = []
    k_values = range(2, 11)   # silhouette score needs k >= 2

    # Compute inertia and silhouette score for each k
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)

        inertia_values.append(model.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    # ---- Plot Elbow ----
    plt.figure(figsize=(6,4))
    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    elbow_plot_path = os.path.join(result_path, "elbow_plot.png")
    plt.savefig(elbow_plot_path)
    plt.close()

    # ---- Plot Silhouette Scores ----
    plt.figure(figsize=(6,4))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    silhouette_plot_path = os.path.join(result_path, "silhouette_plot.png")
    plt.savefig(silhouette_plot_path)
    plt.close()

    print(f"Elbow plot saved at: {elbow_plot_path}")
    print(f"Silhouette plot saved at: {silhouette_plot_path}")

    # ---- Determine best k mathematically ----
    best_index = silhouette_scores.index(max(silhouette_scores))
    best_k = k_values[best_index]

    print(f"\nBest k based on silhouette score = {best_k}")
    return best_k, inertia_values, silhouette_scores
