import torch.nn as nn
import torch

class YOLO_Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20, l_coord=5, l_noobj=0.5, input_size=224, device='cuda') -> None:
        super().__init__()

        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.input_size = input_size
        self.device = device


    def forward(self, y_pred, y_true):

        self.response_mask = y_true[..., 0] #(?, 7, 7)
        gt_box = y_true[..., 1:5] #(?, 7, 7, 4)
        gt_cls = y_true[..., 5:] #(?, 7, 7, 20)
        
        # pred_con = y_pred[..., 0]
        # pred_box = y_pred[..., 1:5]
        # pred_cls = y_pred[..., 5:]

        # gt_box_trans = self.box_trans(gt_box)
        # pred_box_trans = self.box_trans(pred_box)
        # box_iou = self.iou(gt_box_trans, pred_box_trans)
        
        # xy_loss = self.response_mask * torch.sum(torch.square(pred_box[..., 0:2] - gt_box[..., 0:2]), dim=3)
        # xy_loss = torch.sum(xy_loss)
        
        # wh_loss = self.response_mask * torch.sum(torch.square(torch.sqrt(pred_box[..., 2:4] + 1e-5) - torch.sqrt(gt_box[..., 2:4] + 1e-5)), dim=3)
        # wh_loss = torch.sum(wh_loss)
        
        # ob_loss = self.response_mask * torch.square(pred_con - box_iou)
        # ob_loss = torch.sum(ob_loss)
        
        # no_loss = (1 - self.response_mask) * torch.square(pred_con - 0)
        # no_loss = torch.sum(no_loss)
        
        pred_con1 = y_pred[..., 0]
        pred_box1 = y_pred[..., 1:5]
        pred_con2 = y_pred[..., 5]
        pred_box2 = y_pred[..., 6:10]
        pred_cls = y_pred[..., 10:]
        
        gt_box_trans = self.box_trans(gt_box)
        pred_box1_trans = self.box_trans(pred_box1)
        pred_box2_trans = self.box_trans(pred_box2)
        
        box1_iou = self.iou(gt_box_trans, pred_box1_trans)
        box2_iou = self.iou(gt_box_trans, pred_box2_trans)
        best_iou = torch.max(box1_iou, box2_iou)
        
        box1_mask = self.response_mask * (box1_iou == best_iou) #(?, 7, 7)
        box2_mask = self.response_mask * (box2_iou == best_iou)
        
        xy_loss = box1_mask * torch.sum(torch.square(pred_box1[..., 0:2] - gt_box[..., 0:2]), dim=3) + \
            box2_mask * torch.sum(torch.square(pred_box2[..., 0:2] - gt_box[..., 0:2]), dim=3)
        xy_loss = torch.sum(xy_loss)
        
        wh_loss = box1_mask * torch.sum(torch.square(torch.sqrt(pred_box1[..., 2:4] + 1e-5) - torch.sqrt(gt_box[..., 2:4] + 1e-5)), dim=3) + \
            box2_mask * torch.sum(torch.square(torch.sqrt(pred_box2[..., 2:4] + 1e-5) - torch.sqrt(gt_box[..., 2:4] + 1e-5)), dim=3)
        wh_loss = torch.sum(wh_loss)
        
        ob_loss = box1_mask * torch.square(pred_con1 - best_iou) + \
            box2_mask * torch.square(pred_con2 - best_iou)
        ob_loss = torch.sum(ob_loss)
        
        no_loss = (1 - box1_mask) * torch.square(pred_con1 - 0) + \
            (1-box2_mask) * torch.square(pred_con2 - 0)
        no_loss = torch.sum(no_loss)
        
        cls_loss = self.response_mask * torch.sum(torch.square(pred_cls - gt_cls), dim=3)
        cls_loss = torch.sum(cls_loss)
        
        total_loss = self.l_coord * xy_loss + \
            self.l_coord * wh_loss + \
            ob_loss + \
            self.l_noobj * no_loss + \
            cls_loss
        
        return [total_loss/y_pred.shape[0], xy_loss/y_pred.shape[0], wh_loss/y_pred.shape[0], ob_loss/y_pred.shape[0], no_loss/y_pred.shape[0], cls_loss/y_pred.shape[0]]
        
        
        
        
    def box_trans(self, box_ori):
        """

        Args:
            box_ori (tensor): [x, y, w, h] x,y relative to grid, w,h relative to whole image 
        Return:
            box_trans (tensor): [xmin, ymin, xmax, ymax] relative to whole image.
        """
        batch = box_ori.shape[0]
        x, y, w, h = box_ori[..., 0], box_ori[..., 1], box_ori[..., 2], box_ori[..., 3] #(?, 7, 7)
        
        offset_x = torch.tensor([[i for i in range(self.S)]*self.S*batch], device=self.device).view(batch, self.S, self.S) #(?, 7, 7)
        offset_y = torch.transpose(offset_x, dim0=1, dim1=2)
        
        x = (x + offset_x) / self.S
        y = (y + offset_y) / self.S
        
        xmin = x - w / 2.0
        ymin = y - h / 2.0
        xmax = x + w / 2.0
        ymax = y + h / 2.0 
    
        box_trans = torch.stack([xmin, ymin, xmax, ymax], dim=3)
        
        return box_trans
    
    
    
    def iou(self, box1, box2):
        """_summary_

        Args:
            box1 (tensor): [x11, y11, x12, y12]
            box2 (tensor): [x21, y21, x22, y22]
            
            
          11------------------
            |                |
            |                |
            |  21(a)---------------------
            |       |        |          |
            |       |        |          |
            ------------------12(b)     |
                    |                   |
                    |                   |
                    ---------------------22

        """
        
        x11, y11, x12, y12 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3] #(?, 7, 7)
        x21, y21, x22, y22 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        
        xa = torch.max(x11, x21)
        ya = torch.max(y11, y21)
        xb = torch.min(x12, x22)
        yb = torch.min(y12, y22)
        
        w = torch.max(xb-xa, torch.zeros_like(xb))
        h = torch.max(yb-ya, torch.zeros_like(yb))
        
        s1 = (x12-x11)*(y12-y11)
        s2 = (x22-x21)*(y22-y21)
        
        s_inner = w*h
        
        iou = s_inner / (s1 + s2 - s_inner + 1e-5)
        
        
        return iou
        
    
        
        
        
        
        
        



if __name__ == '__main__':
    
    from dataset import MyDataset
    from voc_dataset import VOC_Dataset
    from model import YOLO_Model, YOLO_Tiny
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([224, 224]),
    ])
    
    train_ds = VOC_Dataset(image_root="/home/taozhi/datasets/VOC0712/JPEGImages/",
                        label_root="/home/taozhi/datasets/VOC0712/Annotations/",
                        txt_file="/home/taozhi/datasets/VOC0712/0712trainval.txt",
                        transform=transform)
    image, label = train_ds.__getitem__(0)
    image, label = image.unsqueeze(0), label.unsqueeze(0)
    print(image.shape, label.shape)
    
    model = YOLO_Model(S=7,B=2,C=20,use_voc=True)
    # model = YOLO_Tiny(S=7,B=1,C=20)
    output = model(image)
    
    loss_fn = YOLO_Loss(S=7,B=2,C=20,l_coord=5,l_noobj=1,input_size=448, device='cpu')
    
    loss = loss_fn(output, label)
    print(loss)
        

