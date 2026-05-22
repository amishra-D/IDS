from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================================================
# LOAD PREPROCESSING FILES
# =========================================================

scaler = joblib.load(
    "scaler.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)

# =========================================================
# LOAD ENCODER MODEL
# =========================================================

encoder = load_model(
    "encoder_model.keras"
)

# =========================================================
# LOAD HYBRID XGBOOST MODEL
# =========================================================

hybrid_xgb_model = XGBClassifier()

hybrid_xgb_model.load_model(
    "xgb_model.json"
)

# =========================================================
# LOAD STANDALONE XGBOOST MODEL
# =========================================================

standalone_xgb_model = XGBClassifier()

standalone_xgb_model.load_model(
    "xgb_model_standalone.pkl"
)

# =========================================================
# LOAD CATBOOST MODEL
# =========================================================

catboost_model = CatBoostClassifier()

catboost_model.load_model(
    "catboost_model_standalone.pkl"
)

# =========================================================
# LOAD ADABOOST MODEL
# =========================================================

adaboost_model = joblib.load(
    "adaboost_model_standalone.pkl"
)

print("ALL MODELS LOADED SUCCESSFULLY!")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# =========================================================
# INPUT SCHEMA
# =========================================================

class NetworkData(BaseModel):

    duration: float

    protocol_type: str
    service: str
    flag: str

    src_bytes: float
    dst_bytes: float
    land: float
    wrong_fragment: float
    urgent: float
    hot: float
    num_failed_logins: float
    logged_in: float
    num_compromised: float
    root_shell: float
    su_attempted: float
    num_root: float
    num_file_creations: float
    num_shells: float
    num_access_files: float
    num_outbound_cmds: float
    is_host_login: float
    is_guest_login: float
    count: float
    srv_count: float
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {

        "message": "IDS Multi-Model API Running"

    }

# =========================================================
# TREE MODEL PREPROCESSING
# =========================================================

def preprocess_tree(data):

    input_df = pd.DataFrame(
        [data.dict()]
    )

    # One Hot Encoding
    input_df = pd.get_dummies(
        input_df
    )

    # Match Training Columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_df

# =========================================================
# HYBRID MODEL PREPROCESSING
# =========================================================

def preprocess_hybrid(data):

    input_df = preprocess_tree(
        data
    )

    scaled_input = scaler.transform(
        input_df
    )

    scaled_input = scaled_input.astype(
        np.float32
    )

    return scaled_input

# =========================================================
# GENERIC PREDICTION FUNCTION
# =========================================================

def make_prediction(model, features):

    prediction = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    predicted_label = label_encoder.inverse_transform(
        prediction.astype(int)
    )

    confidence = float(
        np.max(probabilities)
    )

    class_probabilities = {

        label_encoder.inverse_transform([i])[0]:
        round(float(prob), 4)

        for i, prob in enumerate(probabilities[0])

    }

    return {

        "prediction": predicted_label[0],

        "confidence": round(
            confidence,
            4
        ),

        "class_probabilities":
        class_probabilities

    }

# =========================================================
# HYBRID AE + XGBOOST
# =========================================================

@app.post("/predict/hybrid")
def predict_hybrid(data: NetworkData):

    try:

        scaled_input = preprocess_hybrid(
            data
        )

        # Generate Encoded Features
        encoded_features = encoder.predict(
            scaled_input,
            verbose=0
        )

        encoded_features = encoded_features.astype(
            np.float32
        )

        # Feature Fusion
        fused_features = np.concatenate(

            [
                scaled_input,
                encoded_features
            ],

            axis=1
        )

        return make_prediction(

            hybrid_xgb_model,

            fused_features

        )

    except Exception as e:

        return {

            "error": str(e)

        }

# =========================================================
# STANDALONE XGBOOST
# =========================================================

@app.post("/predict/xgboost")
def predict_xgboost(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            standalone_xgb_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/catboost")
def predict_catboost(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            catboost_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/adaboost")
def predict_adaboost(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            adaboost_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }