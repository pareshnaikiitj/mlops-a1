from sklearn.tree import DecisionTreeRegressor
from misc import load_data, get_X_y, split_data, preprocess, train_model, evaluate_model, save_model
import numpy as np

def main():
    df = load_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    model = DecisionTreeRegressor(random_state=42)
    model = train_model(model, X_train_s, y_train)

    mse, preds = evaluate_model(model, X_test_s, y_test)
    print(f"[DecisionTree] Test MSE: {mse:.4f}")

    # optional: compute 5 repeated holdout average (optional - helpful to report)
    # save model
    save_model(model, "decision_tree_model.joblib")

if __name__ == "__main__":
    main()