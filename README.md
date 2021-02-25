# Limes-Germanicus

This repository contains the research done for the Archeology Workgroup Netherlands.
The repository contains the necessary scripts and resources to continue further developement 
of this software.
The software is meant to detect the type of shards that are being found in digsites. The profile
of a shard can be understood by the AI and categorized.

The "lib" folder contains all the processing scripts for the algorithm:
    - NN.py, containing the training code for the AI
    - Prediction.py, the prediction loop
    - progressbar.py, a progressbar
    - image_manipulation>image_processing.py, module for image processing
    - image_manipulation>convolution_filters.py, module convolutional filters

The data folder contains all different types of dataset we've been training on.

The doc folder contains the results of the prediction and the performance of the AI.

The bak folder is not important, containing scripts we created but are not important.