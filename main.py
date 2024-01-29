import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
from fastapi.middleware.cors import CORSMiddleware
import time
from pinecone import Pinecone
import pinecone
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

pc = Pinecone(api_key="dccef309-0bde-44bc-8600-a81312b494c5")
index = pc.Index("imagesearch")
# Your existing functions here...


def display_similar_images(similar_images):
    plt.figure(figsize=(20, 10))
    columns = 5
    rows = len(similar_images) // columns + \
        (1 if len(similar_images) % columns else 0)
    for i, image_path in enumerate(similar_images):
        img = mpimg.imread(image_path)
        plt.subplot(rows, columns, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


# Function to load an image and convert it to a tensor
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
model.eval()


image_dir = 'images'
image_paths = [os.path.join(image_dir, file) for file in os.listdir(
    image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]


def search_similar_images_pinecone(query_image_path, top_k):
    query_image_feature = model(
        process_image(query_image_path)).detach().numpy()
    query_image_feature /= np.linalg.norm(query_image_feature)
    # Convert numpy ndarray to a list
    query_image_feature_list = query_image_feature.flatten().tolist()
    print(query_image_feature)

    # Use the new query method
    query_results = index.query(
        vector=query_image_feature_list, top_k=top_k)
    print(query_results)
    # Check if the response has the expected structure
    if 'matches' in query_results:
        similar_image_ids = [match['id'] for match in query_results['matches']]
    else:
        print("No matches found or unexpected response format.")
        similar_image_ids = []

    return similar_image_ids


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Save or process image here
    image_path = "temp.jpg"
    image.save(image_path)

    start_time = time.time()  # Start time measurement

    # Assuming your search function returns a list of image paths/URLs
    similar_images = search_similar_images_pinecone(image_path, top_k=5)

    end_time = time.time()  # End time measurement
    inference_time = end_time - start_time
    print(inference_time)

    return JSONResponse(content={"similar_images": similar_images})
