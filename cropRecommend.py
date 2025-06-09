from trainedModel import crop_profit_recommendation_with_risk
import pandas as pd
import pickle
import warnings
import numpy as np
def cropProfit(new_sample):

    warnings.filterwarnings("ignore", message="X does not have valid feature names")


    # Load your dataset
    df = pd.read_csv("crop_prices.csv")
    with open('crop_recommendation_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    crops = []

    # Extract components
    model = pipeline['model']
    scaler = pipeline['scaler']
    label_encoder = pipeline['label_encoder']
    feature_columns = pipeline['feature_columns']

    # New sample input (N, P, K, Temperature, Humidity, Ph, Rain)
    new_sample = [90, 42, 43, 20.87, 82.00, 6.5, 200.0]
    new_sample_df = pd.DataFrame([new_sample], columns=feature_columns)
    new_sample_scaled = scaler.transform(new_sample_df)

    # Predict probabilities
    pred_proba = model.predict_proba(new_sample_scaled)
    top_5_indices = np.argsort(pred_proba[0])[::-1][:5]
    top_5_crops = label_encoder.inverse_transform(top_5_indices)

    print("🌾 Top 5 Recommended Crops:")
    for i, crop in enumerate(top_5_crops, 1):
        print(f"{i}. {crop}")

    # Assign predicted crops for market recommendation
    crops = top_5_crops.tolist()


    # Define crops


    # Get all unique markets from Kolkata
    KolkataMarkets = df["Market"].unique().tolist()

    # Load the model pipeline (make sure this file exists and is trained)
    with open(r'crop_recommendation_model.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)

    # Collect recommendations
    recommendations = []

    for market in KolkataMarkets:
        try:
            result = crop_profit_recommendation_with_risk(df, crops, market)
            if result:
                recommendations.extend(result)  # assuming result is a list of dicts
        except Exception as e:
            print(f"Skipping market {market} due to error: {e}")

    # Convert to DataFrame if needed
    recommendations_df = pd.DataFrame(recommendations)

    # Print or save results
    print(recommendations_df.to_json(orient='records', indent=2))
