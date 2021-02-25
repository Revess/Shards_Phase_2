#-------------Imports-------------#
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import time

def core(data_dir, **kwargs):
    batch_size = kwargs.get("batch_size",32)
    img_height = kwargs.get("img_height",256)
    img_width = kwargs.get("img_width",256)
    trainWhere = kwargs.get("gpu/cpu",'gpu')
    modelName = kwargs.get("modelName","Modelx")
    class_names = kwargs.get("class_names",['1','2'])


    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if(trainWhere == 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

    start = time.time()

    if(trainWhere == 'cpu'):
        tf.config.experimental.set_visible_devices(devices=cpus, device_type='CPU')
        tf.debugging.set_log_device_placement(True)
    elif(trainWhere == 'gpu'):
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)
    print("Done loading GPU in: ", str(time.time() - start))

    model = tf.keras.models.load_model(os.path.join('.','data','models',modelName))
    index = 1
    plt.figure(figsize=(8, 8))
    output_np = np.empty([len(os.listdir(data_dir)),len(class_names)])

    for indexs,filee in enumerate(os.listdir(data_dir)):
        im = Image.open(os.path.join(data_dir,filee)).convert('L')
        im = ImageOps.invert(im)
        im = np.expand_dims(np.asarray(im),2)
        img = keras.preprocessing.image.load_img((data_dir + '/' + str(filee)), target_size=(img_height, img_width),color_mode="rgb")
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict_proba(img_array)
        score = np.around(np.array(predictions),2)

        scores = ""
        for i,value in enumerate(score[0]):
            if len(scores) == 0:
                scores = str(class_names[i]) + ": " + "{:.2f}%".format(value) + "\n"
            else:
                scores += str(class_names[i]) + ": " + "{:.2f}%".format(value) + "\n"
            output_np[index-1,i] = value

        print(
            str(filee) + " most likely belongs to {} with percent confidence of: "
            .format(class_names[np.argmax(score)]) + "\n" +scores
        )
        ax = plt.subplot(5,5, index)
        plt.imshow(np.squeeze(im).astype("uint8"), cmap="Greys",interpolation='nearest',vmin=0,vmax=255,)
        plt.title(str("prediction: " + class_names[np.argmax(score)]) + "\n" + 'expected: ' + str(filee) + "\n" + scores)
        plt.axis("off")
        index+=1
    print(output_np)
    np.save("./"+modelName+".npy",output_np)
    plt.show()
