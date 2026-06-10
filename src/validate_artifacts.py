"""Validate that saved model artifacts can be loaded."""

from src.model_loader import load_artifacts


def main():
    model, threshold, feature_list = load_artifacts()

    print("Artifact validation successful.")
    print("Model type:", type(model))
    print("Threshold:", threshold)
    print("Number of features:", len(feature_list))
    print("First 10 features:", feature_list[:10])


if __name__ == "__main__":
    main()
