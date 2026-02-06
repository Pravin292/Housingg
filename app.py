import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD CSS ----------------
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CS file not found. Please ensure style.css exists.")

local_css("house_price_app/style.css")

# ---------------- HEADER ----------------
st.markdown('<h1 style="text-align: center;">🏡 House Price Predictor <span style="font-weight:300; font-size: 0.8em; color: #4facfe;">Pro</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a0a0a0; margin-bottom: 20px;">AI-Powered accurate house price estimation.</p>', unsafe_allow_html=True)

# ---------------- DATA LOADING & TRAINING (CACHED) ----------------
# ---------------- DATA LOADING & TRAINING (CACHED) ----------------
# Remove cache to allow dynamic retraining based on user inputs without complex hash mapping
# Or use st.cache_data for the dataframe only, and separate training.
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("house_prices.csv")
        return df, None
    except FileNotFoundError:
        return None, "File 'house_prices.csv' not found!"

def train_model(df, n_estimators, learning_rate, max_depth):
    if "SalePrice" not in df.columns:
        return None, None, None, "Dataset missing 'SalePrice' column."

    # Preprocessing
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X = pd.get_dummies(X, drop_first=True)
    
    # Train-Test Split (For evaluation metrics only)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Train Evaluation Model (to get honest metrics)
    model_eval = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        objective='reg:squarederror'
    )
    model_eval.fit(X_train, y_train)
    
    # Calculate Metrics
    y_pred = model_eval.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # 2. Train Final Production Model (on FULL dataset for maximum accuracy)
    final_model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    final_model.fit(X, y)
    
    return final_model, X.columns, metrics, None

# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.header("⚙️ Feature Engineering")
    st.write(" tweak the AI parameters to improve accuracy.")
    
    st.subheader("Model Hyperparameters")
    n_estimators = st.slider("Number of Estimators (Trees)", min_value=100, max_value=5000, value=2000, step=100, help="More trees = better learning, but slower.")
    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.5, value=0.1, step=0.01, help="Higher = faster learning, lower = more precision.")
    max_depth = st.slider("Max Depth", min_value=3, max_value=20, value=10, help="Deeper trees capture more complex patterns (and get closer to exact training data).")
    
    if st.button("🔄 Retrain Model", type="primary"):
        st.cache_data.clear() # Clear data cache if needed, though mostly we just rerun
        st.session_state['retrain'] = True

# Load Data
df, error = load_data()
if error:
    st.error(f"❌ {error}")
    st.stop()

# Train Model
with st.spinner(f"Training AI Model with {n_estimators} trees..."):
    model, model_columns, metrics, error = train_model(df, n_estimators, learning_rate, max_depth)

if error:
    st.error(f"❌ {error}")
    st.stop()

# ---------------- MAIN UI LAYOUT ----------------

# Create two tabs: Prediction (Primary) and Insights (Secondary)
tab_predict, tab_insights = st.tabs(["🔮 Estimate Price", "📊 Model Insights"])

# --- TAB 1: PREDICTION ---
with tab_predict:
    st.markdown("### Enter House Details")
    st.markdown("Provide the characteristics of the house to get an instant valuation.")
    
    with st.form("prediction_input_form"):
        # We need to generate inputs for the features. 
        # Since we did one-hot encoding, we have many columns.
        # We should ideally group them.
        
        # Identify numerical vs categorical from the RAW dataframe to make inputs user friendly
        # But the model needs the processed columns.
        # Strategy: Create inputs based on RAW dataframe columns (without SalePrice)
        # Then manually apply get_dummies logic or match the expected columns.
        # FOR ROBUSTNESS in this simple app, we will iterate through the MODEL columns 
        # and provide numeric inputs. For a better UX with categorical, we'd need to map back.
        # Given the instruction "dont change python code", sticking to the notebook's logic 
        # means we feed the model exactly what it expects.
        
        # However, purely numeric inputs for OHE columns (0/1) is ugly.
        # Let's try to infer simple inputs where possible.
        
        user_input_raw = {}
        
        # Layout grid
        cols = st.columns(3)
        
        # Iterate over the original columns (before dummies) to create nice widgets
        input_cols_config = df.drop("SalePrice", axis=1).columns
        
        # We need to construct a single row df from inputs, then get_dummies it exactly like training
        # IMPORTANT: get_dummies on a single row often misses columns present in training.
        # So we must align it to model_columns.
        
        for i, col_name in enumerate(input_cols_config):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    # Number Input
                    avg_val = float(df[col_name].mean())
                    user_input_raw[col_name] = st.number_input(
                        col_name, value=avg_val
                    )
                else:
                    # Select Box for Categorical
                    options = df[col_name].unique().tolist()
                    user_input_raw[col_name] = st.selectbox(
                        col_name, options=options
                    )

        st.markdown("---")
        predict_btn = st.form_submit_button("💰 Predict Price", type="primary")

    if predict_btn:
        # 1. Variables to DataFrame
        input_df_raw = pd.DataFrame([user_input_raw])
        
        # 2. Apply One-Hot Encoding
        input_df_processed = pd.get_dummies(input_df_raw, drop_first=True)
        
        # 3. Align with Model Columns (Add missing columns as 0)
        # Get missing columns in the training test
        missing_cols = set(model_columns) - set(input_df_processed.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            input_df_processed[c] = 0
            
        # Ensure the order of column in the test set is in the same order than in train set
        input_df_processed = input_df_processed[model_columns]

        # 4. Predict
        prediction = model.predict(input_df_processed)[0]
        
        st.success(f"### 🏡 Estimated House Price: ₹ {prediction:,.2f}")
        st.balloons()

# --- TAB 2: INSIGHTS ---
with tab_insights:
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['r2']:.4f}")
    col2.metric("MAE", f"{metrics['mae']:,.2f}")
    col3.metric("RMSE", f"{metrics['rmse']:,.2f}")
    
    st.markdown("---")
    
    st.subheader("Key Price Drivers")
    # Feature Importance Plot
    importance = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    sns.barplot(data=importance, x="Importance", y="Feature", palette="mako", ax=ax)
    ax.set_xlabel("Importance Score", color="#a0a0a0")
    ax.set_ylabel("", color="#a0a0a0")
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#404040")
        
    st.pyplot(fig)
