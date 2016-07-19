import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import re

classes = ["PASperson"]


def exists(path):
	return os.path.exists(path)

def createDir(path):
 	os.system('mkdir %s'%path)

def convert(ImgSizeX,ImgSizeY,m):
	Xmin = int(m[0])
	Ymin = int(m[1])
	Xmax = int(m[2])
	Ymax = int(m[3])
	dw = 1./ImgSizeX
	dh = 1./ImgSizeY
	x = (Xmin + Xmax)/2.0
	y = (Ymin + Ymax)/2.0
	w = Xmax-Xmin
	h = Ymax-Ymin
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	print(Xmin)
	print(Xmax)
	print(Ymin)
	print(Ymax)
	print(ImgSizeX)
	print(ImgSizeY)
	return (x,y,w,h)

def readBBX(root,path,imageId):
	lines  = open('train/annotations/%s.txt'%imageId,'r')
	outfile= open('%s/%s.txt'%(path,imageId),'w')
	ImgSizeX=0
	ImgSizeY=0
	for line in lines:
		if 'Image size' in line:
			m = re.findall(r'(\w*[0-9+]\w*)',line);
			ImgSizeX = int(m[0])
			ImgSizeY = int(m[1])			
		if 'Xmin' in line:
			m = re.findall(r'(\w*[0-9+]\w*)',line.split(':')[1]);
			print(m);
			bb= convert(ImgSizeX,ImgSizeY,m);
			outfile.write("0 " + " ".join([str(a) for a in bb]) + '\n')
	lines.close();
	outfile.close();


def generateLab(root,path,imageId):
	if not(exists(path)):
		createDir(path)
	
	readBBX(root,path,imageId);
	# convertedBox = convert(boxes);
	# writeBBX(path,imageId,convertedBox);


wd = getcwd()
#if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
#    os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
image_ids = open('train/pos.lst').read().strip().split()
list_file = open('train.txt','w')
lablePath = '%s/train/labels'%wd;
os.system('rm -rf %s'%lablePath)
for image_id in image_ids:
    list_file.write('%s/train/images/%s.png\n'%(wd,image_id))
    
    generateLab(wd,lablePath,image_id)

list_file.close()

