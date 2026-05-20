
# main.py — Financial Health Classifier API
# pip install fastapi uvicorn tensorflow scikit-learn joblib
# Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import numpy as np, joblib, json
import tensorflow as tf
from tensorflow import keras


# Custom objects (harus didefinisikan sebelum load model)
class ResidualBlock(keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units; self.dropout_rate = dropout_rate
        from tensorflow.keras import layers
        self.dense1 = layers.Dense(units, use_bias=False)
        self.dense2 = layers.Dense(units, use_bias=False)
        self.bn1    = layers.BatchNormalization()
        self.bn2    = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.relu   = layers.Activation('relu')
        self.add    = layers.Add()

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.add([x, inputs])
        return self.relu(x)

    def get_config(self):
        return {**super().get_config(), 'units': self.units,
                'dropout_rate': self.dropout_rate}


class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma; self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true     = tf.cast(y_true, tf.int32)
        n_classes  = tf.shape(y_pred)[-1]
        y_true_ohe = tf.one_hot(y_true, depth=n_classes)
        y_pred     = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce   = -tf.reduce_sum(y_true_ohe * tf.math.log(y_pred), axis=-1)
        p_t  = tf.reduce_sum(y_true_ohe * y_pred, axis=-1)
        return tf.reduce_mean(self.alpha * tf.pow(1 - p_t, self.gamma) * ce)

    def get_config(self):
        return {**super().get_config(), 'gamma': self.gamma, 'alpha': self.alpha}


CUSTOM_OBJECTS = {'ResidualBlock': ResidualBlock, 'FocalLoss': FocalLoss}

# Load artefak saat startup
MODEL  = keras.models.load_model(
    'financial_health_classifier.keras', custom_objects=CUSTOM_OBJECTS
)
SCALER = joblib.load('scaler.pkl')
with open('feature_cols.json')  as f: FEATURE_COLS  = json.load(f)
with open('label_classes.json') as f: LABEL_CLASSES = json.load(f)

app = FastAPI(title='Financial Health Classifier API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)


class UserFeatures(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    label        : str
    confidence   : float
    probabilities: Dict[str, float]
    savings_rate : float
    expense_ratio: float
    rekomendasi  : str


@app.get('/health')
def health_check():
    return {'status': 'ok', 'model': 'FinancialHealthClassifier v1.0'}


@app.post('/predict', response_model=PredictionResponse)
def predict(user: UserFeatures):
    try:
        X    = np.array([[user.features.get(c, 0.0) for c in FEATURE_COLS]],
                        dtype=np.float32)
        X_sc = SCALER.transform(X)
        probs = MODEL.predict(X_sc, verbose=0)[0]
        pred  = int(np.argmax(probs))
        label = LABEL_CLASSES[pred]
        sr    = user.features.get('savings_rate', 0)
        er    = user.features.get('expense_ratio', 0)

        if label == 'AMAN' and sr >= 0.40:
            rec = 'Kondisi sangat sehat. Pertimbangkan investasi jangka panjang.'
        elif label == 'AMAN':
            rec = 'Kondisi baik. Tingkatkan savings rate hingga 40%.'
        elif label == 'RAWAN':
            rec = 'Perlu perhatian. Buat anggaran bulanan lebih ketat.'
        else:
            rec = 'KRITIS: Evaluasi seluruh pengeluaran dan cari income tambahan.'

        return {
            'label'        : label,
            'confidence'   : round(float(probs[pred]), 4),
            'probabilities': {c: round(float(p), 4)
                              for c, p in zip(LABEL_CLASSES, probs)},
            'savings_rate' : round(float(sr), 4),
            'expense_ratio': round(float(er), 4),
            'rekomendasi'  : rec,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
