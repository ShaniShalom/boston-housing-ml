import data_loader
import explorer
import model_trainer

def run_project():
    print("--- House Price Prediction Project ---")
    
    # 1. טעינה
    df = data_loader.load_modern_housing_data()
    
    # 2. ניתוח ויזואלי
    explorer.plot_correlation_matrix(df)
    
    # 3. אימון המודל
    model, rmse, r2 = model_trainer.train_housing_model(df)
    
    print(f"Model Training Complete.")
    print(f"Accuracy (R2 Score): {r2:.2f}")
    print(f"Average Error: ${rmse:,.2f}")

if __name__ == "__main__":
    run_project()
