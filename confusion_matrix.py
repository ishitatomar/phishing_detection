# Step 10: Plotting Confusion Matrices
for name, metrics in results.items():
    plt.figure(figsize=(4, 4))
    sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Step 11: Comparison Chart
metric_data = pd.DataFrame({
    model: {
        "Accuracy": results[model]["Accuracy"],
        "Precision": results[model]["Precision"],
        "Recall": results[model]["Recall"],
        "F1 Score": results[model]["F1 Score"]
    } for model in results
}).T

metric_data.plot(kind='bar', figsize=(14, 7))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(True)

if __name__ == "__main__":
    
    try:
        df = pd.read_csv("Dataset.csv")
        all_results = run_pipeline(df.copy())  
        
        print("\nFinal Results Summary:")
        for name, metrics in all_results.items():
            print(f"{name}:")
            for metric, value in metrics.items():
                if metric != "Confusion Matrix":
                    print(f"  {metric}: {value:.4f}")
            print()

    except FileNotFoundError:
        print("Error: Dataset.csv not found. Please make sure the file is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
plt.tight_layout()
plt.show()