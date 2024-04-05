from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model('api/disease_prediction.keras')


CLASS_NAMES = ['Cotton_Bacterial_Blight', 'Cotton_Healthy',
               'Cotton_Powdery_Mildew', 'Cotton_Target_spot',
               'Wheat_Leaf_Blight', 'Wheat_healthy', 'Wheat_mildew',
               'maize_ear_rot', 'maize_fall_armyworm', 'maize_stem_borer']


@app.get("/")
async def root():
    return {"message": "Hello World"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
