from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import datasets, transforms, models
import argparse
import utils

# Set details required to train Model
def get_user_input():
    parser = argparse.ArgumentParser(description = 'Set parser for the training network')
    parser.add_argument('--data_dir', 
                         help = 'Data directory - Required', 
                         default = './flowers', 
                         type = str)
    parser.add_argument('--save_dir', 
                         help = 'Directory to Save Model', 
                         default = './', 
                         type = str)
    parser.add_argument('--arch', 
                         help = 'Architecuture to train the model - vgg16/densenet121', 
                         default =  'vgg16', 
                         type = str)
    parser.add_argument('--hidden_layers',
                         help = 'Number of hiddent unit in classifier', 
                         default = [4096,1000], 
                         type = int)
    parser.add_argument('--epochs', 
                         help = 'Number of Epochs', 
                         default = 5, 
                         type = int)
    parser.add_argument('--lr', 
                         help = 'Learning Rate', 
                         default = 0.001, 
                         type = float)
    parser.add_argument('--dropout', 
                         help = 'Drop Out rate', 
                         default = 0.5, 
                         type = float)
    parser.add_argument('--gpu', 
                         help = 'Choose to process in GPU or CPU - If GPU not avilable by default System will take CPU', 
                         default = 'yes', 
                         type = str)

    
    args = parser.parse_args()
    return args


 # Get Data loader and Data Set   
def get_dataset_and_loader(data_dir, model):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225] } 
    
    # Transforms for the training, validation, and testing sets
    data_transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(normalize['mean'], normalize['std'])
                                              ])

    data_transforms_validation = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(normalize['mean'], normalize['std'])
                                              ])

    # Load the datasets with ImageFolder
    data_set = {
                 'train': datasets.ImageFolder(train_dir, transform = data_transforms_training), 
                 'validate': datasets.ImageFolder(valid_dir, transform = data_transforms_validation),
                 'test': datasets.ImageFolder(test_dir, transform = data_transforms_validation)
    }

    # Using the image datasets and the trainforms, define the dataloaders
    loaders = {
                'train': torch.utils.data.DataLoader(data_set['train'], batch_size = 32, shuffle=True),
                'validate': torch.utils.data.DataLoader(data_set['validate'], batch_size = 32, shuffle=True),
                'test': torch.utils.data.DataLoader(data_set['test'], batch_size = 32, shuffle=True)
    }
    
    model.class_to_idx = data_set['train'].class_to_idx

    return loaders


# Build the Model
def build_model(args):
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        #print(model.classifier[0].in_features)
        args.input_size = 25088
    elif args.arch == 'alexnet': 
        model = models.alexnet(pretrained=True)
        #print(model.classifier)
        args.input_size = 9216
    elif args.arch == 'resnet18':
        model = models.alexnet(pretrained=True)
        #print(model.classifier[1].in_features)
        args.input_size = 9216
    else:
        model = models.densenet161(pretrained=True)
        #print(model.classifier)
        args.input_size = 2208

    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier = nn.Sequential(nn.Linear(args.input_size, args.hidden_layers[0]),
                              nn.ReLU(),
                              nn.Dropout(args.dropout),
                              nn.Linear(args.hidden_layers[0], args.hidden_layers[1]),
                              nn.ReLU(),
                              nn.Dropout(args.dropout),
                              nn.Linear(args.hidden_layers[1], 102),
                              nn.LogSoftmax(dim = 1))
   
    return model


# Train the Model
#def train(model, args, loaders, optimizer, criterion):
def train(args):
    model = build_model(args)
    loaders = get_dataset_and_loader(args.data_dir, model)
    optimizer, criterion = utils.get_optimizer_criterion(model, args)
    device = utils.get_device(args)


    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 50
    
    for e in range(args.epochs):
        model.train()
        #print(loaders['train'])
        for inputs, labels in loaders['train']:
            #print(f'{inputs}')
            #print(f'{labels}')
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                test_loss = 0 
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in loaders['validate']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1,dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {e + 1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(loaders['validate']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(loaders['validate']):.3f}")
                running_loss = 0
                model.train()
    accuracy = validate(model, args, loaders, criterion)
    utils.save_model(model, args, optimizer)
    return accuracy      
    
# Vlidate the Model    
def validate(model, args, loaders , criterion):
    accuracy = 0 
    test_loss = 0 
    device = utils.get_device(args)
    model.eval()
         
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model(inputs)
            test_loss += criterion(logps, labels).item()
            
            ps = torch.exp(logps)
            
            top_p, top_class = ps.topk(1, dim = 1)
            
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
   
    print(f"Test accuracy: {accuracy/len(loaders['test']):.3f}")
    accuracy = round(accuracy/len(loaders['test']) * 100 , 2)
    return accuracy 

def main():
    args = get_user_input()
    print('Training Inprogress')
    accuracy = train(args)
    print(f'Training Completed with an Accuracy of {accuracy}% using {args.arch} Architecture')
    
    
if __name__ == "__main__":
    main()
    
    
        
            
            
                        
            
    
    
    


        
