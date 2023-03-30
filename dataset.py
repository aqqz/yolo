from torch.utils.data import Dataset
import torchvision.transforms as T
import os, json, random, torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2



class MyDataset(Dataset):
    def __init__(self, image_root, label_root, txt_file, transform) -> None:
        super().__init__()
        self.image_root = image_root
        self.label_root = label_root
        self.txt_file = txt_file
        self.transform = transform
        
        self.filelist = []
        f = open(self.txt_file, "r+")
        for line in f.readlines():
            self.filelist.append(line.strip())
        f.close()
        
        self.class_list = [
            'cup', 'hand', 'person', 'bottle', 'uav'
        ]    
        
        self.color_list = [ (random.random(), random.random(), random.random()) for i in range(len(self.class_list))]
        
        # yolo param
        self.S = 7
        self.B = 2
        self.C = len(self.class_list)
        self.input_size = 224
    
    
    def __len__(self):
        return len(self.filelist)
    
    
    def __getitem__(self, index):
        image_name = self.filelist[index] + '.jpeg'
        image_path = os.path.join(self.image_root, image_name)
        pil_image = Image.open(image_path).convert('L')
        image_array = np.array(pil_image)
        
        json_name = self.filelist[index] + '.json'
        json_path = os.path.join(self.label_root, json_name)
        image_mark = self.parse_json(json_path)
        # self.draw_mark(image_array, image_mark)
        
        # data augmentation
        if self.txt_file.endswith('train.txt'):
            image_array = self.random_brightness(image_array)
            image_array = self.random_contrast(image_array)
            image_array = self.random_blur(image_array)
            image_array, image_mark = self.random_flip(image_array, image_mark)
            image_array, image_mark = self.random_scale(image_array, image_mark)
            image_array, image_mark = self.random_shift(image_array, image_mark)
        # print(image_mark)
        # self.draw_mark(image_array, image_mark)
        
        if self.transform != None:
            image = self.transform(image_array)
            
        label = self.encode_label(image_mark)
        
        # decode_mark = self.decode_mark(label)
        # image_trans = np.array(torch.permute(image, dims=(1, 2, 0)))
        # print(decode_mark)
        # self.draw_mark(image_trans, decode_mark)
        
        return image, label
        
        
        
        
    
    def parse_json(self, json_path):
        f = open(json_path, "r+")
        
        json_dict = json.load(f)
        height = json_dict['imageHeight']
        width = json_dict['imageWidth']
        shapes = json_dict['shapes']
        
        ob_infos = []
        for shape in shapes:
            ob_name = shape['label']
            ob_id = self.class_list.index(ob_name)
            ob_points = shape['points']
            ob_xmin = ob_points[0][0]
            ob_ymin = ob_points[0][1]
            ob_xmax = ob_points[1][0]
            ob_ymax = ob_points[1][1]

            ob_infos.append([ob_id, ob_xmin, ob_ymin, ob_xmax, ob_ymax])
        
        
        image_mark = [height, width, ob_infos]
        
        return image_mark
    
    
    def draw_mark(self, image_array, image_mark):
    
        h, w, ob_infos = image_mark[0], image_mark[1], image_mark[2]
        
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        
        for ob in ob_infos:
            ob_id, xmin, ymin, xmax, ymax = ob[0], int(ob[1]), int(ob[2]), int(ob[3]), int(ob[4])
            rect = mpatches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=0, color=self.color_list[ob_id])
            ax.add_patch(rect)
            ax.text(xmin-2, ymin-2, s=self.class_list[ob_id], color=self.color_list[ob_id])
            
        plt.show()
        
        
        
    def encode_label(self, image_mark):
        height, width, ob_infos = image_mark[0], image_mark[1], image_mark[2]
        
        label = torch.zeros(size=[self.S, self.S, self.C+5]) #(7, 7, 25)
        
        for ob in ob_infos:
            ob_id, xmin, ymin, xmax, ymax = ob[0], ob[1], ob[2], ob[3], ob[4]
            
            xmin = xmin * self.input_size / width
            ymin = ymin * self.input_size / height
            xmax = xmax * self.input_size / width
            ymax = ymax * self.input_size / height
            
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            offset_x = int(x // (self.input_size / self.S))
            offset_y = int(y // (self.input_size / self.S))
            
            x = x / (self.input_size / self.S) - offset_x
            y = y / (self.input_size / self.S) - offset_y
            w = w / self.input_size
            h = h / self.input_size
            
            label[offset_y][offset_x][0] = 1 #y坐标控制行，x坐标控制列
            label[offset_y][offset_x][1] = x
            label[offset_y][offset_x][2] = y
            label[offset_y][offset_x][3] = w
            label[offset_y][offset_x][4] = h
            
            for k in range(self.C):
                label[offset_y][offset_x][5+k] = 0
                
            label[offset_y][offset_x][5+ob_id] = 1
            
            
        return label
    
    
    def decode_mark(self, label):
        
        ob_infos = []
        
        for i in range(self.S):
            for j in range(self.S):
                grid_vector = label[i, j]
                if grid_vector[0] == 1:
                    x, y, w, h = grid_vector[1].item(), grid_vector[2].item(), grid_vector[3].item(), grid_vector[4].item()
                    
                    x = (x + j) * (self.input_size / self.S)
                    y = (y + i) * (self.input_size / self.S)
                    w = w * self.input_size
                    h = h * self.input_size
                    xmin = x - w / 2.0
                    ymin = y - h / 2.0
                    xmax = x + w / 2.0
                    ymax = y + h / 2.0
                    cls = torch.argmax(grid_vector[5:]).item()
                    ob_infos.append([cls, xmin, ymin, xmax, ymax])
    
        decode_mark = [self.input_size, self.input_size, ob_infos]
        return decode_mark
    
    
    
    def random_contrast(self, image):
        if random.random() < 0.5:
            alpha = random.uniform(0.5, 1.5)
            im = np.uint8(np.clip(image * alpha, 0, 255))
            return im
        return image
    
    
    def random_brightness(self, image):
        if random.random() < 0.5:
            beta = random.randint(-50, 50)
            im = np.uint8(np.clip(image + beta, 0, 255))
            return im
        return image
    
    
    def random_blur(self, image):
        if random.random() < 0.5:
            im = cv2.blur(image, (5, 5))
            return im
        return image
    
    
    def random_flip(self, image, mark):
        if random.random() < 0.5:    
            h, w, ob_infos = mark[0], mark[1], mark[2]
            im = np.fliplr(image).copy()
            for ob in ob_infos:
                xmin = w - ob[3]
                xmax = w - ob[1]
                ob[1] = xmin
                ob[3] = xmax
            return im, mark
        return image, mark
                
                

    def random_scale(self, image, mark):
        if random.random() < 0.5:    
            h, w, ob_infos = mark[0], mark[1], mark[2]
            scale = random.uniform(0.8, 1.2)
            w = int(w * scale)
            im = cv2.resize(image, (w, h))
            
            for ob in ob_infos:
                xmin, xmax = ob[1], ob[3]
                ob[1] = xmin * scale
                ob[3] = xmax * scale
            mark[1] = w
            
            return im, mark
        return image, mark
    
    
    def random_shift(self, image, mark):
        
        h, w, ob_infos = mark[0], mark[1], mark[2]

        shift_x = random.uniform(-w*0.2, w*0.2)
        shift_y = random.uniform(-h*0.2, h*0.2)
    
        if len(image.shape) == 3:
            im = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            im = np.zeros((h, w, 1), dtype=np.uint8)
            image = np.expand_dims(image, axis=2)
            
        
        im[:, :, :] = 128
        
        if shift_x>=0 and shift_y>=0:
            im[int(shift_y):, int(shift_x):, :] = image[:h-int(shift_y), :w-int(shift_x), :]
        elif shift_x>=0 and shift_y<0:
            im[:h+int(shift_y), int(shift_x):, :] = image[-int(shift_y):, :w-int(shift_x), :]
        elif shift_x<0 and shift_y>=0:
            im[int(shift_y):, :w+int(shift_x), :] = image[:h-int(shift_y), -int(shift_x):, :]
        elif shift_x<0 and shift_y<0:
            im[:h+int(shift_y), :w+int(shift_x), :] = image[-int(shift_y):, -int(shift_x):, :]
            
        for ob in ob_infos:
            xmin, ymin, xmax, ymax = ob[1], ob[2], ob[3], ob[4]
            xc = (xmin + xmax) / 2 + shift_x
            yc = (ymin + ymax) / 2 + shift_y
            
            xc_in = (xc > 0) & (xc < w)
            yc_in = (yc > 0) & (yc < h)
            
            box_in = xc_in & yc_in
            if box_in == True:
                ob[1] = xmin + shift_x
                ob[2] = ymin + shift_y
                ob[3] = xmax + shift_x
                ob[4] = ymax + shift_y
            else:
                ob = None
                
        return im, mark
            
            
            
        
            

        
        
        
        

            
        
    
    
    
    
    

if __name__ == '__main__':
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([224, 224]),
    ])
    
    train_ds = MyDataset(image_root='images',
                        label_root='labels',
                        txt_file='train.txt',
                        transform=transform)
    
    val_ds = MyDataset(image_root='images',
                        label_root='labels',
                        txt_file='val.txt',
                        transform=transform)
    
    print(len(train_ds), len(val_ds))
    
    image, label = train_ds.__getitem__(0)
    print(image.shape, label.shape)
    