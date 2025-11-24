import os
import matplotlib.pyplot as plt

def visualize_clusters(features, labels, save_path="results/plots/cluster_visualization.png"):
    """
    Visualizes clusters in a 2D scatter plot (first two feature columns).
    
    Args:
        features (pd.DataFrame or np.ndarray): The feature set used for clustering.
        labels (list or array): Assigned cluster labels.
        save_path (str): Path to save the visualization plot.
    """

    # Make sure results/plots exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))

    # Take only first 2 columns → ideal for Mall_Customers dataset
    x = features[:, 0]
    y = features[:, 1]

    scatter = plt.scatter(x, y, c=labels)   # No color map specified

    plt.title("Customer Segmentation - Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)

    # Save file
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Cluster visualization saved at: {save_path}")
