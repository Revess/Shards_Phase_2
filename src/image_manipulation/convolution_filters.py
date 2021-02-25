from PIL import Image
import numpy as np
import os,sys
import matplotlib.pyplot as plt

def conv2D_fromImage(image,**kwargs):
    #get kwargs
    kernel = kwargs.get("kernel", np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]))
    padding = kwargs.get("padding",False)
    destructive = kwargs.get("destructive",False)
    #Variables
    im = image
    outputMatrix = np.zeros((im.shape[0],im.shape[1]))
    kernelsize=kernel.shape[0]
    kernelmax = 0
    #calculate kernel max
    for row in kernel:
        for number in row:
            kernelmax=kernelmax+(number*255)

    #start code
    #Add padding
    if padding:
        padding = padding*2
        paddingArea = np.zeros((im.shape[0]+padding,im.shape[1]+padding))
        paddingArea[0:im.shape[0],0:im.shape[1]] = im[:,:]
        im = paddingArea
        im = im.astype(np.int64)
    
    #convolve image
    for rown in range(im.shape[0]):
        for coln in range(im.shape[1]):
            if im[rown:rown+kernelsize,coln:coln+kernelsize].shape[0] == kernelsize and im[rown:rown+kernelsize,coln:coln+kernelsize].shape[1] == kernelsize:
                output = im[rown:rown+kernelsize,coln:coln+kernelsize] * kernel
                pixel = 0
                for arrs in output:
                    for numbers in arrs:
                        pixel += numbers
                if pixel > 0:
                    outputMatrix[rown,coln] = pixel
    return outputMatrix

def conv2D_fromPath(imagePath,filename,**kwargs):
    #get kwargs
    kernel = kwargs.get("kernel", np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]))
    padding = kwargs.get("padding",False)
    destructive = kwargs.get("destructive",False)
    #Variables
    outputpath = None
    outputImage = None
    if destructive:
        outputpath = os.path.join(imagePath,filename)
    else:
        if not os.path.exists(imagePath + '\\' + 'convolved'):
            os.mkdir(imagePath + '\\' + 'convolved')
        outputpath = os.path.join(imagePath,'convolved',filename)
    im = Image.open(os.path.join(imagePath,filename)).convert('L')
    im = np.asarray(im)
    outputMatrix = np.zeros((im.shape[0],im.shape[1]))
    kernelsize=kernel.shape[0]
    kernelmax = 0
    #calculate kernel max
    for row in kernel:
        for number in row:
            kernelmax=kernelmax+(number*255)

    #start code
    #Add padding
    if padding:
        padding = padding*2
        paddingArea = np.zeros((im.shape[0]+padding,im.shape[1]+padding))
        paddingArea[0:im.shape[0],0:im.shape[1]] = im[:,:]
        im = paddingArea
        im = im.astype(np.int64)
    
    #convolve image
    for rown in range(im.shape[0]):
        for coln in range(im.shape[1]):
            if im[rown:rown+kernelsize,coln:coln+kernelsize].shape[0] == kernelsize and im[rown:rown+kernelsize,coln:coln+kernelsize].shape[1] == kernelsize:
                output = im[rown:rown+kernelsize,coln:coln+kernelsize] * kernel
                pixel = 0
                for arrs in output:
                    for numbers in arrs:
                        pixel += numbers
                if pixel > 0:
                    outputMatrix[rown,coln] = pixel
    outputMatrix = outputMatrix.astype(np.uint8)
    outputImage = Image.fromarray(outputMatrix)
    outputImage.save(outputpath)

#example input:
# dirt = os.path.join('..','data','datasets','3DShapes','256x256')
# for folder in os.listdir(dirt):
#     for pic in os.listdir(os.path.join(dirt,folder)):
#         kernel = np.array([[-1,-1,-1],[-1,7.8,-1],[-1,-1,-1]])
#         conv2D(os.path.join('..','doc','shard_information','holwerda133-136','Scan_0003.jpg'),pic,kernel=kernel,padding=1)

# kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# conv2D(os.path.join('..','doc','shard_information','holwerda133-136'),'Scan_0003.jpg',kernel=kernel,padding=1)