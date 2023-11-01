from pydantic import BaseModel
from celery_config import execute_pipeline
from blacksheep import Application
import pickle
import os
import json
import base64
import time


app = Application()
post = app.router.post



class order_details(BaseModel):
    order_number: str

@post("/start/")
async def run_pipline(data: order_details):   # Note: Using the Pydantic model directly
    time.sleep(5)
    execute_pipeline.delay(data.dict())
    return {"status": "Order is placed, processing in the background."}
