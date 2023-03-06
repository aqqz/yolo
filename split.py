import os, random

label_root = 'E:\\taozhi\\yolo\\labels'

f1 = open('train.txt', "w+")
f2 = open('val.txt', "w+")

for i in os.listdir(label_root):
    if random.random() < 0.7:
        f1.write(i[:-5] + '\n')
    else:
        f2.write(i[:-5] + '\n')
        
f1.close()
f2.close()