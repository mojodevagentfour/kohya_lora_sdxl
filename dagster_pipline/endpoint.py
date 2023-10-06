from pydantic import BaseModel
from Dagster_pipline import generateuploader_form
from blacksheep import Application
import pickle
import os
import json
import base64


app = Application()
post = app.router.post



class order_details(BaseModel):
    order_number: str

@post("/start/")
async def run_pipline(data: order_details):   # Note: Using the Pydantic model directly
    
    result = generateuploader_form.execute_in_process(run_config={"resources": {"order_data": {"config": data.dict()}}})
    return True
