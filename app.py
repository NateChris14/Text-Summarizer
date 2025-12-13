from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/", tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")

# Running training is slow, blocks server,
# not production safe
@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return {"status": "Training Successful"}
    
    except Exception as e:
        return Response(f"Error Occured! {e}")
    
@app.post("/predict")
async def predict_route(data: TextInput):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(data.text)
        return {"summary": summary}
    except Exception as e:
        raise e
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0",port=8080)