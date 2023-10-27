import argparse
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
import os
import json

# Function to load and preprocess the data
def load_data(data_directory):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'valid', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, dataset_sizes, class_to_idx

# Function to build the model
def build_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg16' or 'resnet50'.")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    return model

# Function to train the model
def train_model(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    dataloaders, dataset_sizes, class_to_idx = load_data(data_directory)
    
    model = build_model(arch, hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/dataset_sizes['train']:.4f}")
    
    model.class_to_idx = class_to_idx
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, save_dir)
    
    print("Training complete. Model checkpoint saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (e.g., vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    print('Directory prints...')
    print(args.data_dir, args.save_dir)
    print('Directory prints...')

    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
