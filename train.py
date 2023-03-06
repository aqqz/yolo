import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from loss import YOLO_Loss
from model import YOLO_Model
from dataset import MyDataset
from voc_dataset import VOC_Dataset
from torch.utils.tensorboard import SummaryWriter
import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150
BATCH_SIZE=32
learning_rate = 1e-3


def train(model: YOLO_Model, train_ds: Dataset, val_ds: Dataset, batch_size, lr, epoch, device='cuda'):
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=18)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=18)
    
    optim = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = YOLO_Loss(S=7,B=2,C=5,l_coord=5,l_noobj=0.5,input_size=224, device=device)
    scheduler = MultiStepLR(optim, [75, 105], gamma=0.1)
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir='runs/' + str(time))
    print(device)
    print(f"training on {len(train_ds)} images, validating on {len(val_ds)} images.")
    
    for t in range(epoch):
        
        # training
        train_loss = 0.0
        model.train()
    
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            
            train_loss += loss.item()
            
            
        # validating
        val_loss = 0.0
        model.eval()

        with torch.no_grad():    
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch: {t+1} train_loss: {train_loss / len(train_dataloader)} val_loss: {val_loss / len(val_dataloader)}")
        writer.add_scalar('Loss/train', train_loss/len(train_dataloader), t+1)
        writer.add_scalar('Loss/val', val_loss/len(val_dataloader), t+1)
        
    
    print('Done!')
    torch.save(model.state_dict(), 'model.pt')
            




if __name__ == '__main__':

    model = YOLO_Model(S=7,B=2,C=20, use_voc=True).to(device)    
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([224, 224]),
    ])
    
    train_ds = VOC_Dataset(image_root="/home/taozhi/datasets/VOC0712/JPEGImages/",
                        label_root="/home/taozhi/datasets/VOC0712/Annotations/",
                        txt_file="/home/taozhi/datasets/VOC0712/0712trainval.txt",
                        transform=transform)
    
    val_ds = VOC_Dataset(image_root="/home/taozhi/datasets/VOC0712/JPEGImages/",
                        label_root="/home/taozhi/datasets/VOC0712/Annotations/",
                        txt_file="/home/taozhi/datasets/VOC0712/07test.txt",
                        transform=transform)
    
    train(model, train_ds, val_ds, batch_size=BATCH_SIZE, lr=learning_rate, epoch=EPOCHS, device=device)
    