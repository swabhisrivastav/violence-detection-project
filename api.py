from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import Model
from PIL import Image
import io
import numpy as np
import base64
import logging
from twilio.rest import Client  
from dotenv import load_dotenv
import os

load_dotenv() 

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")  
client = Client(account_sid, auth_token)

app = FastAPI()
origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = Model()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    logger.info("Received a file for prediction.")
    
    # Log file details
    logger.info(f"File received: {file.filename}, size: {file.size}")
    # Read the file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    logger.info(f"Image shape: {image_np.shape}")

    # Make a prediction using your existing model
    label = model.predict(image=image_np)['label'].title()

    logger.info(f"Predicted label: {label}")

    # Trigger Twilio alert based on the prediction
    alert_message = ""
    if label in ['fight on a street','fire on a street','street violence','Car Crash','violence in office','fire in office']:
        # Set the alert message based on the predicted label
        alert_message = f"{label} alert triggered!"
        
        try:
            message = client.messages.create(
                body=alert_message,
                to='+919108151087',  # Your phone number
                from_='+12537771041'  # Your Twilio phone number
            )
            logger.info(f"Twilio message sent with SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send Twilio message: {e}")
            raise HTTPException(status_code=500, detail="Failed to send alert")

    return {"predicted_label": label, "alert_message": alert_message}