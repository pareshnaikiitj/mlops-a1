from sklearn.kernel_ridge import KernelRidge
from misc import load_data, get_X_y, split_data, preprocess, train_model, evaluate_model, save_model

def main():
    df = load_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    model = KernelRidge(alpha=1.0, kernel='rbf')
    model = train_model(model, X_train_s, y_train)

    mse, preds = evaluate_model(model, X_test_s, y_test)
    print(f"[KernelRidge] Test MSE: {mse:.4f}")

    save_model(model, "kernel_ridge_model.joblib")

if __name__ == "__main__":
    main()
