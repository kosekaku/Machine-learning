import argparse
import torch
from torch import nn
from torchvision import models, transforms
import json
import os

# Function to load the model checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    model = models.__dict__[arch](pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Function to preprocess image
def process_image(image_path):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = image_transform(image)
    
    return image

# Function to perform prediction
def predict(image_path, model, topk, gpu):
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
    
    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk)
    
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = [class_to_idx_inverted[i.item()] for i in top_indices[0]]
    
    return top_probabilities[0].tolist(), top_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--topk', type=int, default=5, help='Number of top K classes to predict')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File containing category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
    top_probabilities, top_classes = predict(args.image_path, model, args.topk, args.gpu)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = [cat_to_name[cls] for cls in top_classes]
    
    for i in range(args.topk):
        print(f"Class: {class_names[i]}, Probability: {top_probabilities[i]:.4f}")
