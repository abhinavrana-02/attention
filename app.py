import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Title of the web app
st.title("🧠 Attention Type Predictor using Eye-Gaze & Physiological Data")

# Upload the CSV file
uploaded_file = st.file_uploader("📤 Upload your test CSV file", type="csv")

if uploaded_file is not None:
    # Load the test data
    test_df = pd.read_csv(uploaded_file)

    # Load the trained model
    model = joblib.load("attention_model.pkl")  # Update path if needed

    # Select the required features
    features = ['Velocity', 'LPD', 'RPD', 'HRV', 'GSR']
    
    if not all(f in test_df.columns for f in features):
        st.error(f"❌ Uploaded file must contain these columns: {features}")
    else:
        X_new = test_df[features]

        # Use the same scaler used during training (you should ideally load it)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_new)

        # Predict
        y_pred_new = model.predict(X_scaled)

        # Reverse map predictions
        label_mapping_reverse = {
            0: 'Sustained Attention',
            1: 'Alternating Attention',
            2: 'Dividing Attention'
        }
        test_df['Predicted_Attention'] = [label_mapping_reverse[p] for p in y_pred_new]

        # Show predictions
        st.subheader("🔍 Sample Predictions")
        st.dataframe(test_df[['Velocity', 'LPD', 'RPD', 'HRV', 'GSR', 'Predicted_Attention']].head())

        # Downloadable output
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, file_name='test_predictions.csv', mime='text/csv')
