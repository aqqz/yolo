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
        self.features = mobilenet_v2(pretrained=True).features
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(7*7*512, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, self.S*self.S*(5*self.B+self.C)),
        #     nn.Sigmoid()
        # )
        self.fc = nn.Sequential(
            nn.Conv2d(1280, 5*self.B+self.C, 1, 1),
            nn.Sigmoid(),
        )
        

    
    def forward(self, x):
        if self.use_voc == False:
            x = self.gray2rgb(x)
        x = self.features(x)
        x = self.fc(x)
        x = x.permute((0, 2, 3, 1)) # use fully conv to local
        return x
    




if __name__ == '__main__':

    import torch

    input = torch.randn(10, 1, 224, 224)
    model = YOLO_Model(S=7,B=2,C=20)
    print(model)

    output = model(input)
    print(output.shape)