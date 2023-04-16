import torch.nn as nn
from torchvision.models import mobilenet_v2

class YOLO_Model(nn.Module):


    def __init__(self, S=7, B=2, C=20, use_voc=False) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.use_voc = use_voc
        
        self.gray2rgb = nn.Conv2d(1, 3, 1, 1)
        self.features = mobilenet_v2(pretrained=True).features[:-1]
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(320, 5*self.B+self.C, 1, 1),
            nn.Sigmoid(),
        )
        
    
    def forward(self, x):
        if self.use_voc == False:
            x = self.gray2rgb(x)
        x = self.features(x)
        x = self.fc(x)
        x = x.permute((0, 2, 3, 1))
        return x
    
    
    
    
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class YOLO_Tiny(nn.Module):
    
    def __init__(self, S=7, B=2, C=20) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            CBL(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            CBL(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            CBL(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            CBL(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            CBL(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            CBL(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            CBL(512, 1024, 3, 1, 1),
        )
        
        self.local = CBL(1024, 256, 3, 1, 1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(7*7*256, S*S*(5*B+C)),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        
        x = self.features(x)
        x = self.local(x)
        x = self.fc(x)
        x = x.view(-1, 7, 7, 25)
        return x
     



if __name__ == '__main__':

    import torch

    input = torch.randn(10, 3, 224, 224)
    model = YOLO_Model(S=7,B=2,C=20,use_voc=True)
    # model = YOLO_Tiny(S=7,B=1,C=20)
    print(model)

    output = model(input)
    print(output.shape)