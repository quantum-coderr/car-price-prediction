import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_next_model_version(model_dir):
    """Scan the model directory and return the next available version number."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1
        
    pattern = re.compile(r"^model_v(\d+)\.pkl$")
    max_version = 0
    
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)
            
    return max_version + 1


def main():
    # 1. Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "cars_data.csv")
    df = pd.read_csv(data_path)

    # 2. Handle missing values
    df['Price'] = df['Price'].fillna(df['Price'].median())
    df['Year'] = df['Year'].fillna(df['Year'].median())
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())

    # 3. Create target and features
    df['PriceLog'] = np.log1p(df['Price'])

    cols_to_drop = ['Id', 'Price', 'PriceLog', 'Vin']
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['PriceLog']

    # 4. Identify categorical and numerical columns
    categorical_columns = ['City', 'State', 'Make', 'Model']
    numeric_columns = ['Year', 'Mileage']

    # 5. Build ColumnTransformer with OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'
    )

    # 6. Build the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # 8. Train the pipeline
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 9. Evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (on Log Price): {mse:.4f}")

    # 10. Save the pipeline with VERSIONING
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    next_version = get_next_model_version(model_dir)
    
    model_filename = f"model_v{next_version}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path} (Version {next_version})")

if __name__ == "__main__":
    main()
