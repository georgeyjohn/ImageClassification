from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import datasets, transforms, models
import argparse
import utils
import json

# Set details required to train Model
def get_user_input():
    parser = argparse.ArgumentParser(description = 'Set parser for the Prediction network')
    parser.add_argument('image_path', 
                         help = 'Image path for prediction - Required', 
                         type = str)
    parser.add_argument('--model_path', 
                         help = 'Directory to Save Model', 
                         default = 'checkpoint.pth', 
                         type = str)
    parser.add_argument('--category_names',
                         help = 'Flower Category', 
                         default= './cat_to_name.json', 
                         type = str)
    parser.add_argument('--top_k', 
                         help = 'Top K Class', 
                         default = 5, 
                         type = int)
    parser.add_argument('--gpu', 
                         help = 'Choose to process in GPU or CPU - If GPU not avilable by default System will take CPU', 
                         default = 'yes', 
                         type = str)
    
    args = parser.parse_args()
    return args

#Get tensor Image
def process_image(image_path):
    image = Image.open(image_path)  
    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225] } 

    inference_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(normalize['mean'], normalize['std'])])
    
    return inference_transforms(image)

# This function will take the image and Make predection
def predict(args):
    device = utils.get_device(args)
    model, check_point = utils.load_checkpoint(args.model_path)
    
    image_tensor = process_image(args.image_path)
    #inference_transforms(image)
    
    top_num = args.top_k
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        model_input = image_tensor.unsqueeze(0)        
        model.to(device)
        logps = model(model_input)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_num, dim=1)

        idx_to_class = {val: key for key, val in    
                                          model.class_to_idx.items()}
        
        top_labels=[]
        for c in top_class.cpu().numpy().tolist()[0]:
            top_labels.append(idx_to_class[c])
        
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
        
    print(f'Here are the top {top_num} Predictions....\n')
    for i in range(top_num):
         print(f'Flower Name: {top_flowers[i].title()} \nProbability: {str(round(float(top_p[0][i] * 100),2))}%\n')
    return

def main():
    args = get_user_input()
    predict(args)
    
if __name__ == '__main__':
    main()
    