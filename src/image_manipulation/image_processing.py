from PIL import Image
import numpy as np
from math import sin, cos
import os
from os.path import join,isdir,exists,basename,normpath,splitext
import random

def readImage(path,grey=True):
    '''
    Read image
    Args:
    -path: path to read from
    -grey: output should be greyscale or not
    '''
    image = Image.open(path)
    if grey:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    image = np.copy(np.asarray(image))
    return image

def writeImage(image,path,mode="png"):
    '''
    Writes image
    Args:
    -image: image data to write
    -path: path to write to
    '''
    if mode == "png":   
        path = path[:-3]+"png"
    if mode == "jpg":
        path = path[:-3]+"jpg"
    Image.fromarray(image.astype(np.uint8)).save(path)

def trueBW(image=None, image_path=None, output_path=None):
    '''
    Sets all values in the image to either 0 or 255
    Args:
    -image: pass a 2d numpy matrix as an image.
    -image_path: read the image from a given path, if given with the argument "image" it uses that matrix instead.
    -output_path: the path to save the image to, None will make the function return the image.
    '''
    if image is None and image_path is not None:
        image = readImage(image_path,grey=True)
    for rown in range(image.shape[0]):
        for coln in range(image.shape[1]):
            if image[rown,coln] > 127:
                image[rown,coln] = 255
            elif image[rown,coln] < 128:
                image[rown,coln] = 0
    if output_path is None:
        return image
    else:
        writeImage(image,output_path)

def convertFileType(image=None, image_path=None, output_path=None):
    if image is None and image_path is not None:
        image = readImage(image_path,grey=False)
    file_ = splitext(basename(image_path))
    file_ = file_[0] + ".png"
    output_path = output_path + "\\" + file_
    if output_path is None:
        return image
    else:
        writeImage(image,output_path)

def warpImage(image=None, image_path=None, output_path=None,rndboundries=[4,8],wacky=False):
    '''
    Distorts the given image
    Args:
    -image: pass a 2d numpy matrix as an image.
    -image_path: read the image from a given path, if given with the argument "image" it uses that matrix instead.
    -output_path: the path to save the image to, None will make the function return the image.
    -rndboundries: the boundries the randomizer will go over, default to 4 to 12.
    '''
    if image is None and image_path is not None:
        image = readImage(image_path,grey=True)
    rows, cols = image.shape
    img_output = np.full(image.shape, 255 ,dtype=image.dtype)
    distortionx = random.randint(rndboundries[0], rndboundries[1])
    distortiony = random.randint(rndboundries[0], rndboundries[1])
    choise = random.randint(0,1)
    for i in range(rows):
        for j in range(cols):
            if wacky:
                chance = random.randint(0,100)
                if chance < 30:
                    offset_x = int(distortionx * sin(2 * 3.14 * i / 150))
                    offset_y = int(distortiony * cos(2 * 3.14 * j / 150))
                elif chance >= 30 and chance < 50:
                    offset_x = int(distortionx * sin(2.05 * 3.14 * i / 150))
                    offset_y = int(distortiony * cos(2.05 * 3.14 * j / 150))
                else:
                    rndnumber1 = random.randint(15,55)
                    rndnumber2 = random.randint(15,55)
                    while rndnumber1 > rndnumber2:
                        rndnumber1 = random.randint(15,55)
                        rndnumber2 = random.randint(15,55)
                    distortionx = random.randint(rndboundries[0], rndboundries[1])
                    distortiony = random.randint(rndboundries[0], rndboundries[1])
                    offset_x = int(distortionx * sin(2 * 3.14 * i / 150))
                    offset_y = int(distortiony * cos(2 * 3.14 * j / 150))
            else:
                offset_x = int(distortionx * sin(2 * 3.14 * i / 150))
                offset_y = int(distortiony * cos(2 * 3.14 * j / 150))
            if i+offset_y < rows and j+offset_x < cols:
                if choise:
                    img_output[i,j] = image[(i+offset_y)%rows,(j+offset_x)%cols]
                else:
                    img_output[i,j] = image[abs(i-offset_y)%rows,abs(j-offset_x)%cols]
            else:
                img_output[i,j] = 255
    if output_path is None:
        if wacky:
            return (img_output*0.9) + (0.1*np.random.normal(128,128,img_output.shape))
        else:
            return img_output
    else:
        if wacky:
            writeImage((img_output*0.9) + (0.1*np.random.normal(128,128,img_output.shape)),output_path)
        else:
            writeImage(img_output,output_path)
            
def addNoise(image=None, image_path=None, output_path=None, strength=0.015):
    '''
    Adds noise to an image
    Args:
    -image: pass a 2d numpy matrix as an image.
    -image_path: read the image from a given path, if given with the argument "image" it uses that matrix instead.
    -output_path: the path to save the image to, None will make the function return the image.
    '''
    if image is None and image_path is not None:
        image = readImage(image_path,grey=True)
    mean = np.average(np.reshape(image,(image.shape[0]*image.shape[1])))
    stdv = np.std(np.reshape(image,(image.shape[0]*image.shape[1])))
    img_output = ((1-strength)*image) + (strength*np.random.normal(mean,stdv,image.shape))
    if output_path is None:
        return img_output
    else:
        writeImage(img_output,output_path) 

def resizeImage(image=None, image_path=None, output_path=None,max_height=None,max_width=None,squared=False):
    '''
    Resizes images in an image in the ratio of the expected output.
    This function is supposed te be used before adding padding, in
    in order to make all objects have approximately have a similair 
    size.
    Args:
    -image: pass a 2d numpy matrix as an image.
    -image_path: read the image from a given path, if given with the argument "image" it uses that matrix instead.
    -output_path: the path to save the image to, None will make the function return the image.
    -max_height: the maximum height of the image set
    -max_width: the maximum width of the image set
    '''
    if image is None and image_path is not None:
        image = readImage(image_path,grey=True)
    height,width,depth = image.shape
    if not squared:
        if max_height is not None and max_width is not None:
            if height == max_height or width == max_width:
                image = image.astype(np.uint8)
                image = Image.fromarray(image) 
                return image
            height *= max_width/image.shape[1]
            width *= max_width/image.shape[1]
            if height > max_height:
                width *= max_height/height
                height *= max_height/height
            if width > max_width:
                height *= max_width/width
                width *= max_width/width
    else:
        height = max_height
        width = max_width
    image = image.astype(np.uint8)
    image = Image.fromarray(image)    
    image_resized = image.resize((round(width),round(height)), Image.ANTIALIAS)
    if output_path is None:
        return np.asarray(image_resized)
    else:
        image_resized.save(output_path)

def findImages(data_dir):
    '''
    A function to find how many directories with images there are in the given directory.
    Also finds out the maximum width and height of the images that are found.
    Args:
    data_dir: a string that contains the directory of the data with it's subfolders.
    '''
    image_dirs = []
    height,width = 0,0
    for file_ in os.listdir(data_dir): 
        if str(file_) == "output":
            continue
        elif ".png" in str(file_):
            image = Image.open(join(data_dir,file_)).convert('L')
            image = np.asarray(image)
            if image.shape[0] > height:
                height,width = image.shape
        elif isdir(join(data_dir,file_)):
            image_dirs.append(join(data_dir,file_))
            height,width,x = findImages(join(data_dir,file_))
            del x
        else:
            print("Error only accepting .png images")
    return height,width,image_dirs
    
def addPadding(image=None, image_path=None, output_path=None, max_width=0, max_height=0, boundry_offset=5, squared=True):
    '''
    Adds a padding layer around the images
    Args:
    -image: pass a 2d numpy matrix as an image.
    -image_path: read the image from a given path, if given with the argument "image" it uses that matrix instead.
    -output_path: the path to save the image to, None will make the function return the image.
    -max_width: the maximum width of the image set
    -max_height: the maximum width of the image set
    -boundry_offset: size of the border around the image
    -squared: weather image should be squared, otherwise image will be rectangular
    '''
    if image is None and image_path is not None:
        image = readImage(image_path,grey=True)
    max_width+=(boundry_offset*2)
    max_height+=(boundry_offset*2)
    image = np.asarray(image)
    if squared:
        if max_height > max_width:
            max_width = max_height
        elif max_width > max_height:
            max_height = max_width
    padded_image = np.full((max_height,max_width),255)
    offsetW = round((max_width-image.shape[1])/2)
    offsetH = round((max_height-image.shape[0])/2)
    padded_image[offsetH:offsetH+image.shape[0],
    offsetW:offsetW+image.shape[1]] = image[:,:]
    if output_path is None:
        return padded_image
    else:
        writeImage(padded_image,output_path)

def rescaleDataset(data_dir, output_dir=None, add_padding=False):
    '''
    Rescales the entire dataset
    Args:
    -data_dir: path to the dataset.
    -output_dir: path to output the new dataset. If none, will make an output folder
    -add_padding: add padding to the image.
    '''
    height, width, directories = findImages(data_dir)
    if output_dir is None:
        if not exists(join(data_dir, 'output')):
            os.mkdir(join(data_dir, 'output'))
        output_dir = join(data_dir,'output')
    for folders in directories:
        if not exists(join(output_dir, basename(normpath(folders)))):
            os.mkdir(join(output_dir, basename(normpath(folders))))
        for file_ in os.listdir(folders):
            image = readImage(join(folders,file_),grey=True)
            image = resizeImage(image=image,max_height=height,max_width=width)
            if add_padding:
                image = addPadding(image=image,max_width=width,max_height=height)
            writeImage(image,join(output_dir,basename(normpath(folders)),file_))

def resizeDataset(data_dir,output_dir=None,dims=[128,128],squared=False):
    '''
    Rescales the entire dataset to a size
    Args:
    -data_dir: path to the dataset.
    -output_dir: path to output the new dataset. Can be none
    -dims: dimensions to scale to.
    '''
    height, width, directories = findImages(data_dir)
    if len(directories) == 0:
        directories = [data_dir]
    if output_dir is None:
        if not exists(join(data_dir, 'output')):
            os.mkdir(join(data_dir, 'output'))
        output_dir = join(data_dir,'output')
    for folders in directories:
        if not exists(join(output_dir, basename(normpath(folders)))):
            os.mkdir(join(output_dir, basename(normpath(folders))))
        for file_ in os.listdir(folders):
            if file_ != "output":
                image = readImage(join(folders,file_),grey=False)
                resizeImage(image=image,output_path=join(output_dir,basename(normpath(folders)),file_),max_height=dims[0],max_width=dims[1],squared=squared)
