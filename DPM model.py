import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
sns.set_style("whitegrid")

# Constants
MODEL_SAVE_PATH = "saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def feature_engineering(df):
    df = df.copy()
    if 'Supply Level' in df.columns and 'Demand Level' in df.columns:
        df['Supply_Demand_Ratio'] = df['Supply Level'] / (df['Demand Level'] + 1e-5)
    if 'Month' in df.columns:
        df['Is_Peak_Season'] = df['Month'].isin([3,4,5,10,11]).astype(int)
    return df

# Load or Train Model Function
def train_and_save_model(df, model_name="random_forest"):
    # Preprocessing
    df= feature_engineering(df)
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Price':  # Don't encode target
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Features and Target
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    if model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_name == "xgboost":
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    else:
        model = LinearRegression()

    # Training
    model.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(model, f"{MODEL_SAVE_PATH}/price_model.pkl")
    joblib.dump(label_encoders, f"{MODEL_SAVE_PATH}/label_encoders.pkl")
    joblib.dump(X.columns, f"{MODEL_SAVE_PATH}/feature_columns.pkl")

    return model, X_test, y_test


# Prediction Function
def predict_new_data(model, new_df, label_encoders, feature_columns):
    # Preprocess new data same as training
    new_df = feature_engineering(new_df)
    if 'price' in new_df.columns:
        new_df.drop(columns=['price'], inplace=True)

    for col, le in label_encoders.items():
        if col in new_df.columns:
            known_classes = list(le.classes_)
            new_df[col] = new_df[col].apply(lambda x: x if x in known_classes else 'Unknown')

            # Update LabelEncoder to include 'Unknown'
            if 'Unknown' not in known_classes:
                le.classes_ = np.append(le.classes_, 'Unknown')

            new_df[col] = le.transform(new_df[col])
    predictions = model.predict(new_df)

    return predictions


# Main App
st.title("Dynamic Price Prediction System")

# Section 1: Model Training
st.header("Step 1: Train Model on Base Dataset")

base_files = st.file_uploader("Upload multiple datasets (CSV)", type="csv", accept_multiple_files=True, key="train_data")

if base_files:
    dfs = [pd.read_csv(file) for file in base_files]
    train_df = pd.concat(dfs, ignore_index=True)
    st.success(f"Loaded {len(train_df)} records from {len(base_files)} files.")

    model_type = st.selectbox("Select Model Type",
                              ["Random Forest", "XGBoost", "Linear Regression"])

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, X_test, y_test = train_and_save_model(
                train_df,
                model_name=model_type.lower().replace(" ", "_")
            )

        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        st.success(f"Model trained successfully! R2 Score: {r2:.4f}")

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)
            fig, ax = plt.subplots()
            feat_importances.sort_values().plot(kind='barh', ax=ax)
            st.pyplot(fig)

# Section 2: Making Predictions
st.header("Step 2: Predict on New Data")
pred_file = st.file_uploader("Upload NEW data for prediction (CSV)", type="csv", key="predict_data")

if pred_file and os.path.exists(f"{MODEL_SAVE_PATH}/price_model.pkl"):
    # Load artifacts
    model = joblib.load(f"{MODEL_SAVE_PATH}/price_model.pkl")
    label_encoders = joblib.load(f"{MODEL_SAVE_PATH}/label_encoders.pkl")
    feature_columns = joblib.load(f"{MODEL_SAVE_PATH}/feature_columns.pkl")

    # Processing new data
    new_df = pd.read_csv(pred_file)
    predictions = predict_new_data(model, new_df, label_encoders, feature_columns)

    # Show results
    result_df = new_df.copy()
    result_df['Predicted_Price'] = predictions.round(2)

    st.subheader("Prediction Results")
    st.dataframe(result_df)

    # Download results
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Predictions",
        csv,
        "vegetable_price_predictions.csv",
        "text/csv"
    )
elif pred_file and not os.path.exists(f"{MODEL_SAVE_PATH}/price_model.pkl"):
    st.error("Please train a model first before making predictions!")