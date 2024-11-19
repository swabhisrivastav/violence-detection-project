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
from collections import deque
import time

load_dotenv()

TIME_WINDOW = 3 
DETECTION_THRESHOLD = 10  
ALERT_COOLDOWN = 10  

detection_times = deque()
last_alert_time = 0

def check_for_alert(detection_times, last_alert_time,label):
    current_time = time.time()
    
    while detection_times and current_time - detection_times[0] > TIME_WINDOW:
        detection_times.popleft()

    if len(detection_times) >= DETECTION_THRESHOLD:
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            print(f"Alert: {label} detected!")
            logger.info(f"Alert: {label} detected!")
            return current_time  
    
    return last_alert_time

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

model = Model()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    last_alert_time = 0
    current_time = time.time()
    label_to_display = ""
    logger.info("Received a file for prediction.")
    
    # Log file details
    logger.info(f"File received: {file.filename}, size: {file.size}")
    # Read the file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    logger.info(f"Image shape: {image_np.shape}")

    labels = model.predict(image=image_np)['label'].title()
    for label_dict in labels:
        label = label_dict['label'].lower()

        if any(keyword in label for keyword in ['violence', 'fight', 'fire', 'crash']):
            detection_times.append(current_time)
            last_alert_time = check_for_alert(detection_times, last_alert_time)
            if len(detection_times) >= DETECTION_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                logger.info(f"Predicted label: {label}")
                last_alert_time = current_time
            label_to_display = label.capitalize()  
        else:
            detection_times.clear()


    alert_message = ""
    if label_to_display in ['fight on a street','fire on a street','street violence','Car Crash','violence in office','fire in office']:
        alert_message = f"{label_to_display} alert triggered!"
        
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

    return {"predicted_label": label_to_display, "alert_message": alert_message}