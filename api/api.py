import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

# Load the model and vectorizer
model = joblib.load('sms_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

threshold = 0.9

# Create the FastAPI app
app = FastAPI()

# Define the input data format
class SMSInput(BaseModel):
    text: str
    
class SMSBatchInput(BaseModel):
    messages: List[SMSInput]

# Define the prediction endpoint
@app.post("/predict-message-type")
def predict_message_type(data: SMSInput):
    sms_vectorized = vectorizer.transform([data.text])
    # Predict Probabilities
    probabilities = model.predict_proba(sms_vectorized)
    max_prob = np.max(probabilities)
    classified_type = model.predict(sms_vectorized)[0]
    type = 'unknown'
    if max_prob >= threshold:
        type = classified_type
    return {"type": type}

@app.post("/predict-messages-type")
def predict_messages_type(data: SMSBatchInput):
    sms_texts = [sms.text for sms in data.messages]
    sms_vectorized = vectorizer.transform(sms_texts)
    predictions = model.predict(sms_vectorized)
    probabilities = model.predict_proba(sms_vectorized)
    # result = [{"sms": sms_text, "type": type} 
    #           for sms_text, pred in zip(sms_texts, predictions)]
    
    # Map predictions to types
    result = []
    for sms_text, pred, prob in zip(sms_texts, predictions, probabilities):
        max_prob = np.max(prob)  # Get the maximum probability
        type = "unknown"
        
        # Check the threshold
        if max_prob >= threshold:
            type =pred
        
        # Append the result
        result.append({
            "text": sms_text,
            "type": type,
            "probability": max_prob
        })
    
    return result
