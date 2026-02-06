# 🏠 House Price Predictor Pro

A high-performance house price estimation tool built with **XGBoost** and **Streamlit**. This application allows users to predict house prices interactively or explore model performance insights with a modern, glassmorphic UI.

## 🚀 Features
- **Instant Prediction**: Enter house characteristics (Square feet, Year built, etc.) to get a real-time price estimate in ₹.
- **Dynamic Training Lab**: Adjust AI hyperparameters (Number of trees, Learning rate, Max depth) directly from the sidebar.
- **Model Insights**: Interactive visualizations of feature importance and performance metrics (R², MAE, RMSE).
- **Premium UI**: Modern dark-mode interface with glassmorphism and gradient accents.

## 🛠️ Tech Stack
- **AI/ML**: XGBoost Regressor
- **Data**: Pandas, NumPy
- **App**: Streamlit
- **Visuals**: Matplotlib, Seaborn
- **Styling**: Custom CSS (Vanilla)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pravin292/Housingg.git
   cd Housingg
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run house_price_app/app.py
   ```

## 📊 Dataset
The model is trained on the provided `house_prices.csv`, utilizing advanced regression techniques to capture complex price drivers like living area, quality indicators, and year of construction.

---
*Created by [Pravin](https://github.com/Pravin292)*