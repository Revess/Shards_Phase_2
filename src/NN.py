import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import tensorflow as tf

def set_enviroment(gpu_cpu="gpu"):
    if(gpu_cpu == 'cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    if(gpu_cpu == 'cpu'):
        tf.config.experimental.set_visible_devices(devices=cpus, device_type='CPU')
        tf.debugging.set_log_device_placement(False)
    elif(gpu_cpu == 'gpu'):
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus, 'GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu,True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)

def create_model(input_shape,num_classes):
    return tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=input_shape),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(input_shape[1]),
        tf.keras.layers.Dense(num_classes)
    ])

def load_dataset(path,image_size,subset,color_mode,batch_size):
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            validation_split=0.2,
            subset=subset,
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode="categorical"
        )
    except RuntimeError as e:
        print("images not in same size: ", e)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    class_names = dataset.class_names
    if subset == "training":
        dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    elif subset == "validation":
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, class_names

def core(data_dir=None,
        batch_size=16,
        epochs=50,
        input_shape=(120,213,3),
        color_mode="rgb",
        gpu_cpu="gpu",
        model_name="ModelrealShards_480p"):
    set_enviroment(gpu_cpu)
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    #Create a dataset for training
    train_ds, class_names = load_dataset(data_dir,[input_shape[0],input_shape[1]],"training",color_mode,batch_size)
    val_ds = load_dataset(data_dir,[input_shape[0],input_shape[1]],"validation",color_mode,batch_size)[0]
    print(class_names)

    with mirrored_strategy.scope():
        model = create_model(input_shape,len(class_names))

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save(os.path.join('.','data','models',model_name))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()