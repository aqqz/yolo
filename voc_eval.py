# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
 
import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
 
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
 
    return objects
 
def voc_ap(rec, prec, use_07_metric=False):   # voc2007的计算方式和voc2012的计算方式不同，目前一般采用第二种
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
 
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
 
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
 
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
 
 
## 程序入口
 
def voc_eval(detpath,           # 保存检测到的目标框的文件路径，每一类的目标框单独保存在一个文件
             annopath,          # Annotations的路径
             imagesetfile,      # 测试图片名字列表
             classname,         # 类别名称
             cachedir,          # 缓存文件夹
             ovthresh=0.5,      # IoU阈值
             use_07_metric=False):      # mAP计算方法
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
 
    # first load gt   获取真实目标框
    # 当程序第一次运行时，会读取Annotations下的xml文件获取每张图片中真实的目标框
    # 然后把获取的结果保存在annotations_cache文件夹中
    # 以后再次运行时直接从缓存文件夹中读取真实目标
 
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
 
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
 
    # extract gt objects for this class 提取该类的真实目标
    class_recs = {}
    npos = 0     #保存该类一共有多少真实目标
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # 保存名字为imagename的图片中，类别为classname的目标框的信息
        bbox = np.array([x['bbox'] for x in R])  #目标框的坐标
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  #是否是难以识别的目标
        det = [False] * len(R)       #每一个目标框对应一个det[i]，用来判断该目标框是否已经处理过
        npos = npos + sum(~difficult)    #计算总的目标个数
        class_recs[imagename] = {'bbox': bbox,              # 把每一张图像中的目标框信息放到class_recs中
                                 'difficult': difficult,
                                 'det': det}
 
    # read dets
    detfile = detpath.format(classname)     # 打开classname类别检测到的目标框文件
    with open(detfile, 'r') as f:
        lines = f.readlines()
 
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]           # 图像名字
    confidence = np.array([float(x[1]) for x in splitlines])   # 置信度
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])   # 目标框坐标
 
    # sort by confidence  按照置信度排序
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
 
    # go down dets and mark TPs and FPs
    nd = len(image_ids)       # 统计检测到的目标框个数
    tp = np.zeros(nd)         # 创建tp列表，列表长度为目标框个数
    fp = np.zeros(nd)         # 创建fp列表，列表长度为目标框个数
 
    for d in range(nd):
        R = class_recs[image_ids[d]]   # 得到图像名字为image_ids[d]真实的目标框信息
        bb = BB[d, :].astype(float)    # 得到图像名字为image_ids[d]检测的目标框坐标
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float) # 得到图像名字为image_ids[d]真实的目标框坐标
 
        if BBGT.size > 0:
            # compute overlaps  计算IoU
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
 
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
 
            overlaps = inters / uni
            ovmax = np.max(overlaps)     # 检测到的目标框可能预若干个真实目标框都有交集，选择其中交集最大的
            jmax = np.argmax(overlaps)
 
        if ovmax > ovthresh:   # IoU是否大于阈值
            if not R['difficult'][jmax]:   # 真实目标框是否难以识别
                if not R['det'][jmax]:     # 该真实目标框是否已经统计过
                    tp[d] = 1.             # 将tp对应第d个位置变成1
                    R['det'][jmax] = 1     # 将该真实目标框做标记
                else:
                    fp[d] = 1.     # 否则将fp对应的位置变为1
        else:
            fp[d] = 1.     # 否则将fp对应的位置变为1
 
    # compute precision recall
    fp = np.cumsum(fp)        # 按列累加，最大值即为tp数量
    tp = np.cumsum(tp)        # 按列累加，最大值即为fp数量
    rec = tp / float(npos)    # 计算recall
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)   # 计算精度
    ap = voc_ap(rec, prec, use_07_metric)   # 计算ap
 
    return rec, prec, ap
