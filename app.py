# Load model directly
from fastapi import FastAPI, File, UploadFile
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import uvicorn
import io
import os
# pip install python-multipart

app = FastAPI()

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")


@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Prepare inputs and generate caption
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return {"caption": caption}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
