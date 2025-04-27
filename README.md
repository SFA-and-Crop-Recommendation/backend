# Crop Recommendation and Price Prediction

This project helps recommend the top 5 crops based on soil and weather conditions and predicts their market prices after 6 months using machine learning models.

---

## 💡 Features
- **Crop Recommendation**: Using XGBoost classifier
- **Crop Price Forecasting**: Using LSTM model

---

## 🚀 Setup Instructions

### 1. Install Required Libraries
Before running the project, install these Python libraries:

```bash
pip install pandas
pip install scikit-learn
pip install tensorflow   # (Make sure your Python version is 3.10 or lower)
pip install xgboost
pip install seaborn
pip install matplotlib
```

> **Note:**
> - TensorFlow latest versions may not support Python 3.11+. It is recommended to use **Python 3.8 to 3.10**.
> - You can create a virtual environment if needed.

### 2. Required Files
- `crop_recommendation_model.pkl` : Pre-trained model for crop recommendation.
- `train_model.py` : Contains functions for fetching market data and training LSTM.
- Your dataset file if needed (`your_crop_data.csv`)

---

## 📁 Project Structure
```bash
|— main.py (your main script)
|— crop_recommendation_model.pkl
|— train_model.py
|— README.md
|— requirements.txt (optional)
```

---

## 🔄 Git Instructions (Quick Start)

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-link>
git push -u origin main
```

If you need to update later:
```bash
git add .
git commit -m "Updated project"
git push
```

To pull latest code:
```bash
git pull origin main
```

---

## 🚀 Future Improvements
- Merge both models into a complete pipeline.
- Add a front-end UI.
- Dockerize the application.

---

# 👋 Happy Coding!


