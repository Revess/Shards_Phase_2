import os, argparse
import numpy as np
from lib.NN import core
from lib.Prediction import core as predCore
from lib.image_manipulation.image_processing import resizeDataset

parser = argparse.ArgumentParser(description='To train and predict data. Also dont forget to create a dataset',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_dir', metavar='PATH', type=str, required=True, help='help')
parser.add_argument('--original_data', metavar='PATH', type=str, required=True, help='help')
parser.add_argument('--image_shape', nargs='+', metavar='IMAGE_SHAPE', type=int, default=[128,128,3], help='help')
parser.add_argument('--color_mode', metavar='COLOR_MODE', type=str, default='rgb', help='help')
parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=50, help='help')
parser.add_argument('--batch_size', metavar='BATCH_SIZE', type=int, default=32, help='help')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='model_x', help='help')
parser.add_argument('--train', metavar='TRAIN', type=bool, default=False,  help='help')
parser.add_argument('--predict', metavar='PREDICT', type=bool, default=False,  help='help')
parser.add_argument('--create_dataset', metavar='CREATE_DATASET', type=bool, default=False,  help='help')
parser.add_argument('--class_names', metavar='CLASS_NAMES', nargs='+', default=["1","2"],  help='help')

def main():
    args = parser.parse_args()
    if args.create_dataset:
        resizeDataset(args.original_data,args.dataset_dir,args.image_shape[:2],squared=True)

    if args.train:
        core(data_dir=args.dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            input_shape=tuple(args.image_shape),
            color_mode=args.color_mode,
            gpu_cpu="gpu",
            model_name=args.model_name)

    if args.predict:
        predCore(args.dataset_dir,
            modelName=args.model_name,
            img_height=int(args.image_shape[0]),
            img_width=int(args.image_shape[1]),
            class_names = args.class_names)

if __name__ == '__main__':
    main()