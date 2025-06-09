# import warnings
# import pandas as pd
# import numpy as np
# import pickle
# from trainedModel import crop_profit_recommendation_with_risk

# def cropProfit(new_sample):
#     warnings.filterwarnings("ignore", message="X does not have valid feature names")

#     # Load dataset and pretrained model pipeline
#     df = pd.read_csv("crop_prices.csv")
#     with open('crop_recommendation_model.pkl', 'rb') as f:
#         pipeline = pickle.load(f)

#     model = pipeline['model']
#     scaler = pipeline['scaler']
#     label_encoder = pipeline['label_encoder']
#     feature_columns = pipeline['feature_columns']

#     # Prepare input sample
#     new_sample_df = pd.DataFrame([new_sample], columns=feature_columns)
#     new_sample_scaled = scaler.transform(new_sample_df)
#     pred_proba = model.predict_proba(new_sample_scaled)[0]

#     # Select crops with >50% probability
#     valid_indices = np.where(pred_proba > 0.5)[0]
#     if len(valid_indices) == 0:
#         print("No crops with probability > 50% found.")
#         return []

#     sorted_indices = valid_indices[np.argsort(pred_proba[valid_indices])[::-1]]
#     top_5_indices = sorted_indices[:5]
#     top_5_crops = label_encoder.inverse_transform(top_5_indices)

#     crops = top_5_crops.tolist()

#     # If only one crop, just return its name
#     if len(crops) == 1:
#         print(crops[0])
#         return crops

#     # Get all Kolkata markets from dataset
#     KolkataMarkets = df["Market"].unique().tolist()

#     recommendations = []
#     for market in KolkataMarkets:
#         try:
#             result = crop_profit_recommendation_with_risk(df, crops, market)
#             if result:
#                 recommendations.extend(result)
#         except Exception as e:
#             print(f"Skipping market {market} due to error: {e}")

#     # Handle output: only crop names expected
#     if isinstance(recommendations, list) and all(isinstance(crop, str) for crop in recommendations):
#         print(recommendations)
#         return recommendations
#     elif isinstance(recommendations, list) and all(isinstance(crop, dict) and "Crop" in crop for crop in recommendations):
#         crop_names = list({r["Crop"] for r in recommendations})
#         print(crop_names)
#         return crop_names
#     else:
#         print("Unexpected format in recommendations.")
#         return []


# import json
# import warnings
# import pandas as pd
# import numpy as np
# import pickle
# import sys
# from trainedModel import crop_profit_recommendation_with_risk

# def cropProfit(new_sample):
#     try:
#         warnings.filterwarnings("ignore", message="X does not have valid feature names")

#         # Load dataset and pretrained model pipeline
#         df = pd.read_csv("crop_prices.csv")
#         with open('crop_recommendation_model.pkl', 'rb') as f:
#             pipeline = pickle.load(f)

#         model = pipeline['model']
#         scaler = pipeline['scaler']
#         label_encoder = pipeline['label_encoder']
#         feature_columns = pipeline['feature_columns']

#         # Prepare input sample
#         new_sample_df = pd.DataFrame([new_sample], columns=feature_columns)
#         new_sample_scaled = scaler.transform(new_sample_df)
#         pred_proba = model.predict_proba(new_sample_scaled)[0]

#         # Select crops with >50% probability
#         valid_indices = np.where(pred_proba > 0.5)[0]
#         if len(valid_indices) == 0:
#             return {"Success": False, "error": "No crops with probability > 50% found."}

#         sorted_indices = valid_indices[np.argsort(pred_proba[valid_indices])[::-1]]
#         top_5_indices = sorted_indices[:5]
#         top_5_crops = label_encoder.inverse_transform(top_5_indices)
#         crops = top_5_crops.tolist()

#         # Get all Kolkata markets from dataset
#         KolkataMarkets = df["Market"].unique().tolist()
#         recommendations = []

#         for market in KolkataMarkets:
#             try:
#                 result = crop_profit_recommendation_with_risk(df, crops, market)
#                 if result:
#                     recommendations.extend(result)
#             except Exception:
#                 continue  # silently skip markets that fail

#         if isinstance(recommendations, list) and all(isinstance(crop, str) for crop in recommendations):
#             return {"Success": True, "crops": recommendations}
#         elif isinstance(recommendations, list) and all(isinstance(crop, dict) and "Crop" in crop for crop in recommendations):
#             crop_names = list({r["Crop"] for r in recommendations})
#             return {"Success": True, "crops": crop_names}
#         else:
#             return {"Success": False, "error": "Unexpected format in recommendations."}

#     except Exception as e:
#         return {"Success": False, "error": str(e)}


# if __name__ == "__main__":
#     try:
#         if len(sys.argv) != 2:
#             print(json.dumps({"error": "Expected one argument: JSON input"}))
#             sys.exit(1)

#         input_data = json.loads(sys.argv[1])
#         result = cropProfit(input_data)
#         print(json.dumps(result))

#     except Exception as e:
#         print(json.dumps({"error": str(e)}))
#         sys.exit(1)

import json
import warnings
import pandas as pd
import numpy as np
import pickle
import sys
from trainedModel import crop_profit_recommendation_with_risk

def cropProfit(new_sample):
    try:
        warnings.filterwarnings("ignore", message="X does not have valid feature names")

        # Load dataset and pretrained model pipeline
        df = pd.read_csv("crop_prices.csv")
        with open('crop_recommendation_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)

        model = pipeline['model']
        scaler = pipeline['scaler']
        label_encoder = pipeline['label_encoder']
        feature_columns = pipeline['feature_columns']

        # Prepare input sample
        new_sample_df = pd.DataFrame([new_sample], columns=feature_columns)
        new_sample_scaled = scaler.transform(new_sample_df)
        pred_proba = model.predict_proba(new_sample_scaled)[0]

        # Select crops with >50% probability
        valid_indices = np.where(pred_proba > 0.5)[0]
        if len(valid_indices) == 0:
            return {"Success": False, "error": "No crops with probability > 50% found."}

        sorted_indices = valid_indices[np.argsort(pred_proba[valid_indices])[::-1]]
        top_5_indices = sorted_indices[:5]
        top_5_crops = label_encoder.inverse_transform(top_5_indices)
        crops = top_5_crops.tolist()

        # Get all Kolkata markets from dataset
        KolkataMarkets = df["Market"].unique().tolist()
        recommendations = []

        for market in KolkataMarkets:
            try:
                result = crop_profit_recommendation_with_risk(df, crops, market)
                if isinstance(result, list):
                    recommendations.extend(result)
                elif isinstance(result, str):
                    recommendations.append(result)
            except Exception:
                continue  # silently skip markets that fail

        # Process recommendations into unique crop names
        if isinstance(recommendations, list) and all(isinstance(crop, str) for crop in recommendations):
            final_crops = list(set(recommendations))
        elif isinstance(recommendations, list) and all(isinstance(crop, dict) and "Crop" in crop for crop in recommendations):
            final_crops = list({r["Crop"] for r in recommendations})
        else:
            return {"Success": False, "error": "Unexpected format in recommendations."}

        # Build dictionary with keys crop1, crop2, ...
        crop_result = {f"crop{i+1}": crop for i, crop in enumerate(final_crops)}
        return crop_result

    except Exception as e:
        return {"Success": False, "error": str(e)}


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print(json.dumps({"error": "Expected one argument: JSON input"}))
            sys.exit(1)

        input_data = json.loads(sys.argv[1])
        result = cropProfit(input_data)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
