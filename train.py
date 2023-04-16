import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ConstantLR, SequentialLR
from loss import YOLO_Loss
from model import YOLO_Model, YOLO_Tiny
from dataset import MyDataset
from voc_dataset import VOC_Dataset
from torch.utils.tensorboard import SummaryWriter
import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 95
BATCH_SIZE= 32
learning_rate = 1e-2


def train(model: YOLO_Model, train_ds: Dataset, val_ds: Dataset, batch_size, lr, epoch, device):
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    
    optim = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optim = Adam(model.parameters())
    loss_fn = YOLO_Loss(S=7,B=2,C=20,l_coord=5,l_noobj=0.5,input_size=224, device=device)
    scheduler1 = ConstantLR(optim, factor=0.1, total_iters=5)
    scheduler2 = ConstantLR(optim, factor=1.0, total_iters=30)
    scheduler3 = ConstantLR(optim, factor=0.1, total_iters=30)
    scheduler4 = ConstantLR(optim, factor=0.01, total_iters=30)
    scheduler = SequentialLR(optim, [scheduler1, scheduler2, scheduler3, scheduler4], milestones=[5, 35, 65])
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir='runs/' + str(time))
    print(device)
    print(f"training on {len(train_ds)} images, validating on {len(val_ds)} images.")
    
    for t in range(epoch):
        
        # training
        train_loss = 0.0
        xy, wh, ob, no, cl = 0.0, 0.0, 0.0, 0.0, 0.0
        model.train()
    
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss[0].backward()
            optim.step()
            
            train_loss += loss[0].item()
            xy += loss[1].item()
            wh += loss[2].item()
            ob += loss[3].item()
            no += loss[4].item()
            cl += loss[5].item()
            
            
        # validating
        val_loss = 0.0
        _xy, _wh, _ob, _no, _cl = 0.0, 0.0, 0.0, 0.0, 0.0
        model.eval()

        with torch.no_grad():    
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss[0].item()
                _xy += loss[1].item()
                _wh += loss[2].item()
                _ob += loss[3].item()
                _no += loss[4].item()
                _cl += loss[5].item()
        
        scheduler.step()
        print(f"Epoch: {t+1} train_loss: {train_loss / len(train_dataloader)} val_loss: {val_loss / len(val_dataloader)}")
        
        writer.add_scalars('totalloss', {
            'train': train_loss/len(train_dataloader),
            'val': val_loss/len(val_dataloader)
        }, t+1)
        writer.add_scalars('xyloss', {
            'train': xy/len(train_dataloader),
            'val': _xy/len(val_dataloader)
        }, t+1)
        writer.add_scalars('whloss', {
            'train': wh/len(train_dataloader),
            'val': _wh/len(val_dataloader)
        }, t+1)
        writer.add_scalars('obloss', {
            'train': ob/len(train_dataloader),
            'val': _ob/len(val_dataloader)
        }, t+1)
        writer.add_scalars('noloss', {
            'train': no/len(train_dataloader),
            'val': _no/len(val_dataloader)
        }, t+1)
        writer.add_scalars('clsloss', {
            'train': cl/len(train_dataloader),
            'val': _cl/len(val_dataloader)
        }, t+1)
        
    
    print('Done!')
    torch.save(model.state_dict(), 'model.pt')
            




if __name__ == '__main__':

    model = YOLO_Model(S=7,B=2,C=20,use_voc=True).to(device)   
    # model = YOLO_Tiny(S=7,B=1,C=20).to(device) 
    
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
    
    # train_ds = MyDataset(image_root='images',
    #                      label_root='labels',
    #                      txt_file='train.txt',
    #                      transform=transform)
    
    # val_ds = MyDataset(image_root='images',
    #                    label_root='labels',
    #                    txt_file='val.txt',
    #                    transform=transform)
    
    train(model, train_ds, val_ds, batch_size=BATCH_SIZE, lr=learning_rate, epoch=EPOCHS, device=device)
    