import numpy as np
from PIL import Image
import csv
import ast
import glob
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

import mmcv
from typing import List

__title__ = 'Annotation Converter'
__version__ = '3.0.0'
__author__ = 'Kim, Huijo'


def load_file(args):
    ''' load folder location for train  '''
    train_files = [os.path.normpath(i) for i in sorted(glob.glob(r'input/images/*'))]
    ann_files = [os.path.normpath(i) for i in sorted(glob.glob(r'input/anns/*'))]
    ann_save_loc = r'output'
            
    return train_files, ann_files, ann_save_loc

def obj_class(obj_type: str, args):  
    ''' Convert object type(name) --> index '''
    with open(args.obj_class) as json_file:  # open configure file
        conv = json.load(json_file)

    return conv[obj_type]

def read_annotation(annotation: str, args):  
    ''' Open annotations file and sort'''
    annotations = []
    with open(annotation, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            annotations.append(row)

    ''' Sort the array based on the class '''
    annotations = sorted(np.array(annotations[1:]), key=lambda d: obj_class(ast.literal_eval(d[6])['class'], args)) # If you setup different name, change class to something else.
    return np.array(annotations)

def extract_polygons(annotations: np.ndarray):  
    ''' Extract polygons from annotation of VIA format '''
    polys = []
    xcords = [None]*len(annotations)
    ycords = [None]*len(annotations)
    obj_types = []

    ''' For the polygon points of all object in the annotations file '''
    for i_row in range(0, len(annotations)):
        polys.append([])
        shape = ast.literal_eval(annotations[i_row][5]) # polygon? rect?
        class_name = ast.literal_eval(annotations[i_row][6])['class'] # Symbol? Black?

        if shape['name'] == 'polygon':
            xcords[i_row] = shape['all_points_x']
            ycords[i_row] = shape['all_points_y']
            obj_types.append(class_name)  # get the class of the polygon

        elif shape['name'] == 'rect':
            xcords[i_row] = [shape['x']-shape['width'], shape['x']-shape['width'],
                         shape['x']+shape['width'], shape['x']+shape['width']]
            ycords[i_row] = [shape['y']-shape['height'], shape['y']+shape['height'],
                         shape['y']+shape['height'], shape['y']-shape['height']]
            obj_types.append(class_name)  # get the class of the polygon

        for j_point in range(len(xcords[i_row])):  # each pixel of the polygon
            ''' Pair the x and y coordinate '''
            polys[i_row].append((xcords[i_row][j_point], ycords[i_row][j_point]))

    return polys, obj_types

def draw_mask(i: int, polygons: List, obj_num: int, segmentation: np.ndarray, syn_mode: bool):
    ''' Generate annotation mask for original image '''
    segmentation_mask = np.array(polygons[i]) # polygon of i_th object in an image
    cv2.fillPoly(segmentation, np.int32([segmentation_mask.reshape((-1, 1, 2))]), obj_num)

    return segmentation

def convert(args, train_files: List, ann_files: List, ann_save_loc: str):  # load file and conver it

    ''' i_th image file is loaded from directory folder. '''
    for i in range(len(train_files)):

        img_name = train_files[i]
        print(i, 'th image , ', img_name)
        im_array = np.array(Image.open(img_name))
        annotations = read_annotation(ann_files[i], args) # read i_th image's annotations

        ''' Object's polygons with its type of object are obtained. '''
        object_polygons, obj_type = extract_polygons(annotations)
        segmentation = np.zeros((im_array.shape[0], im_array.shape[1]))
        num_object = len(annotations) # How many objects in an image

        ''' Generate original annotation (segmentation) '''
        for j_object in range(num_object): 

            ''' Call every jth object '''
            obj_num = int(obj_class(obj_type[j_object], args))
            segmentation = draw_mask(j_object, object_polygons, obj_num, segmentation, syn_mode=False)

        if args.save:
            ''' Only generate segmentation maps (VGG IA --> mmsegmentation custom dataset) '''
            converted_ann_map = Image.fromarray(segmentation.astype(np.uint8)).convert('P')
            '''train_path or val_files'''
            converted_ann_map.save(ann_save_loc+r'/'+Path(train_files[i]).stem+'.png')

        else:             
            print('debug mode. new file is not generated. If generation is neeeded, add --save')
            plt.subplot(1, 2, 1)
            plt.gca().set_title('Mask')
            plt.imshow(segmentation, cmap='gist_gray')
            plt.subplot(1, 2, 2)
            plt.gca().set_title('Image')
            plt.imshow(im_array)
            plt.show()

    print("Process is done!")

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Convert VIA dataset into Customdataset for mmsegmentation")
    parser.add_argument("--obj_class", default='setup/conv.json',
                        help="dictionaries of object class. More import object is listed in higher value")
    parser.add_argument("--save", action='store_true', default=None,
                        help="Save the image files. ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    train_files, ann_files,  ann_save_loc = load_file(args)
    convert(args, train_files, ann_files, ann_save_loc)

