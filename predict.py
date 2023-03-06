import torch, os
from dataset import MyDataset
from voc_dataset import VOC_Dataset
from model import YOLO_Model
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class YOLO_Predictor():
    
    def __init__(self, model, test_ds: MyDataset, score_threshold=0.5, iou_threshold=0.5, device='cpu') -> None:
        self.model = model
        self.test_ds = test_ds
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.S = self.test_ds.S
        self.B = self.test_ds.B
        self.C = self.test_ds.C
        self.input_size = self.test_ds.input_size
        self.class_list = self.test_ds.class_list
        self.color_list = self.test_ds.color_list
    
    
    def predict(self, index):
        
        if type(self.test_ds) == MyDataset:
            image_name = self.test_ds.filelist[index] + '.jpeg'
        else:
            image_name = self.test_ds.filelist[index] + '.jpg'
        image_path = os.path.join(self.test_ds.image_root, image_name)
        pil_image = Image.open(image_path)
        image_array = np.array(pil_image)
        height, width = image_array.shape[0], image_array.shape[1]
        
        self.model.eval()
        image, _ = self.test_ds.__getitem__(index)
        
        input = image.unsqueeze(0)
        output = self.model(input)
        ob_infos = self.post_progress(output[0], height, width)
        print(ob_infos)
        
        self.draw(image_array, ob_infos)
        
        
        
    
    def post_progress(self, output_tensor, ori_h, ori_w):
        
        box_before = []
        
        for i in range(self.S):
            for j in range(self.S):
                grid_vector = output_tensor[i, j]
                box1_con = grid_vector[0].item()
                box1_x = grid_vector[1].item()
                box1_y = grid_vector[2].item()
                box1_w = grid_vector[3].item()
                box1_h = grid_vector[4].item()
                box2_con = grid_vector[5].item()
                box2_x = grid_vector[6].item()
                box2_y = grid_vector[7].item()
                box2_w = grid_vector[8].item()
                box2_h = grid_vector[9].item()
                box_cls = grid_vector[10:]
                
                box_id = torch.argmax(box_cls).item()
                box1_score = box1_con * box_cls[box_id].item()
                box2_score = box2_con * box_cls[box_id].item()
                
                if box1_score >= self.score_threshold:
                    box1_xmin, box1_ymin, box1_xmax, box1_ymax = self.box_trans(box1_x, box1_y, box1_w, box1_h, j, i, ori_h, ori_w)
                    box_before.append([box_id, box1_score, box1_xmin, box1_ymin, box1_xmax, box1_ymax])
                
                if box2_score >= self.score_threshold:
                    box2_xmin, box2_ymin, box2_xmax, box2_ymax = self.box_trans(box2_x, box2_y, box2_w, box2_h, j, i, ori_h, ori_w)
                    box_before.append([box_id, box2_score, box2_xmin, box2_ymin, box2_xmax, box2_ymax])
                    
              
        box_after = self.nms(box_before)
        
        return box_after
        
            
                
                
    def box_trans(self, x, y, w, h, offset_x, offset_y, ori_h, ori_w):
        """box trans from [x, y, w, h] to [xmin, ymin, xmax, ymax] real coord

        Args:
            x (_type_): _description_
            y (_type_): _description_
            w (_type_): _description_
            h (_type_): _description_
            offset_x (_type_): _description_
            offset_y (_type_): _description_
        """
        
        x = ( x + offset_x ) / self.S * self.input_size
        y = ( y + offset_y ) / self.S * self.input_size
        w = w * self.input_size
        h = h * self.input_size
        
        x = x * ori_w / self.input_size
        y = y * ori_h / self.input_size
        w = w * ori_w / self.input_size
        h = h * ori_h / self.input_size
        
        xmin = x - w / 2.0
        ymin = y - h / 2.0
        xmax = x + w / 2.0
        ymax = y + h / 2.0
        
        return xmin, ymin, xmax, ymax
                
                    
                    
    def nms(self, box_before: list):
        
        box_after = []
        box_before.sort(key=lambda x:x[1], reverse=True)
        
        while len(box_before) > 0:
            box_after.insert(0, box_before[0])
            box_before.remove(box_before[0])
            
            for box in box_before:
                test_iou = self.iou(box[2:], box_after[0][2:])
                if test_iou >= self.iou_threshold:
                    box_before.remove(box)
        
        return box_after
        
                   
                
        
        
    def iou(self, box1, box2):
        """compute iou with box1 an box2

        Args:
            box1 (list): [x11, y11, x12, y12]
            box2 (list): [x21, y21, x22, y22]
        """
    
        x11, y11, x12, y12 = box1[0], box1[1], box1[2], box1[3]
        x21, y21, x22, y22 = box2[0], box2[1], box2[2], box2[3]
        
        xa = max(x11, x21)
        ya = max(y11, y21)
        xb = min(x12, x22)
        yb = min(y12, y22)
        
        w = max(xb-xa, 0)
        h = max(yb-ya, 0)
        
        s1 = (x12-x11)*(y12-y11)
        s2 = (x22-x21)*(y22-y21)
        s_inner = w*h
        
        iou = s_inner / (s1 + s2 - s_inner + 1e-5)
    
        return iou
    
    
    
    
    def draw(self, image_array, ob_infos):
        
        fig, ax = plt.subplots()
        ax.imshow(image_array)
        
        for ob in ob_infos:
            cls, score, xmin, ymin, xmax, ymax = ob[0], ob[1], int(ob[2]), int(ob[3]), int(ob[4]), int(ob[5])
            
            rect = mpatches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=0, color=self.color_list[cls])
            ax.add_patch(rect)
            ax.text(xmin-2, ymin-2, s=self.class_list[cls] + ': ' + str(score)[:5], color=self.color_list[cls])
        
        plt.show()
    
    
    
    
    


if __name__ == '__main__':
    
    model = YOLO_Model(S=7,B=2,C=20, use_voc=True)
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([224, 224]),
    ])
    
    val_ds = VOC_Dataset(image_root="/home/taozhi/datasets/VOC0712/JPEGImages/",
                        label_root="/home/taozhi/datasets/VOC0712/Annotations/",
                        txt_file="/home/taozhi/datasets/VOC0712/07test.txt",
                        transform=transform)
    
    
    predictor = YOLO_Predictor(model, val_ds, score_threshold=0.5, iou_threshold=0.5, device='cpu')
    predictor.predict(5)