from src.data_preprocessing import preprocess_data
from src.best_k import find_best_k
from src.train_model import train_model
from src.visualize import visualize_clusters

def main():
    print("Preprocessing data...")
    processed_path = "data/processed/processed_data.csv"
    preprocess_data(processed_path=processed_path)   # Save CSV

    print("Finding best k...")
    best_k, elbow_fig, silhouette_fig = find_best_k(processed_path)

    print(f"Best K found: {best_k}")

    print("Training model...")
    model_path, labels, features = train_model(best_k, processed_path)

    print("Training complete. Model saved at:", model_path)

    #OPTIONAL
    print("Generating cluster visualization...")
    visualize_clusters(features, labels)

if __name__ == "__main__":
    main()
