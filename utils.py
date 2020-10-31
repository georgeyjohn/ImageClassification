import torch
from torch import optim
from torch import nn

# Get GPU/CPU is available
def get_device(args):
    if args.gpu == 'yes':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return'cpu'

# Get Opti,izer Criterion
def get_optimizer_criterion(model, args):
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    return optimizer, criterion

def save_model(model, args, optimizer):

    checkpoint = {'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'features': model.features,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'model': model.cpu(), 
                 }
   
    torch.save(checkpoint, str(args.save_dir + 'checkpoint.pth'))
    print('Flower Model saved succesfully')
    return

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']
 