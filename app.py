import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0083B8;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('model_churn_v2.pkl', 'rb'))
    scaler = pickle.load(open('scaler_v2.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# --- HEADER ---
st.title("🤖 Telco Customer Churn Intelligence")
st.markdown("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["👤 Single Customer Prediction", "📂 Bulk Prediction (CSV)", "📈 Model Insights"])

with tab1:
    # ... (existing code for Single Customer Prediction)
    col1, col2 = st.columns([1, 1])
    # ...

    with col1:
        st.subheader("Input Customer Data")
        with st.container():
            tenure = st.slider("Tenure (Months)", 0, 72, 12, help="How many months the customer has stayed with the company")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
            
            predict_btn = st.button("Analyze Status", key="single_predict")

    with col2:
        st.subheader("Analysis Result")
        if predict_btn:
            # Prepare data
            input_data = np.array([[tenure, monthly_charges, total_charges]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            # Display
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK: CHURN")
                st.metric("Churn Probability", f"{probability*100:.1f}%")
                st.progress(float(probability))
                st.warning("**Recommendation:** Offer loyalty rewards or contract extension discounts immediately.")
            else:
                st.success("### ✅ LOW RISK: LOYAL")
                st.metric("Churn Probability", f"{(probability)*100:.1f}%")
                st.progress(float(probability))
                st.info("**Recommendation:** Maintain current service quality and monitor satisfaction.")

with tab2:
    st.subheader("Upload Batch Data for Prediction")
    st.write("Upload a CSV file containing columns: `tenure`, `MonthlyCharges`, and `TotalCharges`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            
            # Validation
            required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            if all(col in df_input.columns for col in required_cols):
                
                # Preprocessing
                df_clean = df_input[required_cols].copy()
                df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
                
                # Predict
                X_scaled = scaler.transform(df_clean)
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1]
                
                # Add to dataframe
                df_input['Prediction'] = np.where(preds == 1, 'Churn', 'Loyal')
                df_input['Probability (%)'] = (probs * 100).round(2)
                
                # Show results
                st.success(f"Successfully processed {len(df_input)} records!")
                st.dataframe(df_input, use_container_width=True)
                
                # Summary
                churn_count = (preds == 1).sum()
                st.write(f"📊 **Summary:** Detected **{churn_count}** high-risk customers out of {len(df_input)}.")
                
                # Download
                csv = df_input.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv',
                )
            else:
                st.error(f"Missing columns! Ensure your CSV has: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.subheader("Key Factors Driving Churn")
    st.markdown("""
    This chart shows which features are most important in our model's decision-making process. 
    Higher values indicate a stronger influence on the prediction results.
    """)
    
    # Extract feature importance
    try:
        importances = model.feature_importances_
        feature_names = ['Tenure', 'Monthly Charges', 'Total Charges']
        
        # Create a DataFrame for the chart
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Display Bar Chart
        st.bar_chart(fi_df.set_index('Feature'))
        
        # Explanations
        st.markdown("### 💡 What do these results mean?")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.info("**Tenure**\n\nThe length of time a customer has stayed. Usually, newer customers are at higher risk.")
        
        with col_exp2:
            st.info("**Monthly Charges**\n\nThe monthly bill amount. High monthly costs often lead to price-sensitive churn.")
            
        with col_exp3:
            st.info("**Total Charges**\n\nThe total amount paid. This reflects long-term loyalty and contract value.")
            
    except Exception as e:
        st.warning(f"Feature importance not available for this model type: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed with ❤️ for Data Learning Portfolio")
