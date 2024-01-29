import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from pinecone import Pinecone
pc = Pinecone(api_key="dccef309-0bde-44bc-8600-a81312b494c5")
index = pc.Index("imagesearch")


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

image_dir = '/Users/alibiserikbay/Developer/pinecone_search/images'
image_paths = [os.path.join(image_dir, file) for file in os.listdir(
    image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
# Process and upload images
# Process and upload images
# Process images and print the shape of the generated feature vectors
for path in image_paths:
    with torch.no_grad():
        img_tensor = process_image(path)
        feature = model(img_tensor)
        normalized_feature = feature.squeeze().numpy() / np.linalg.norm(feature)

        # Print the shape of the feature vector

        # Convert numpy ndarray to a list
        normalized_feature_list = normalized_feature.tolist()
        index.upsert(vectors=[(path, normalized_feature_list)])
        print(
            f"Feature vector shape is added {path}: {normalized_feature.shape}")


# Function to search similar images in Pinecone
