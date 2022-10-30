import argparse
import os

from final_model import *


parser = argparse.ArgumentParser(description='please provide path to labeled folders and path to '
                                             'folder with unlabeled images')
parser.add_argument('--labeled_folders', type=str, required=False, default=None,
                    help='input folder with labeled data')
parser.add_argument('--folder_to_classify',  type=str, required=True,
                    help='folder with images to classify')
parser.add_argument('--model_folder',  type=str, required=True,
                    help='folder to save model checkpoints')

args = parser.parse_args()

if args.labeled_folders is not None:
    if not os.path.exists(args.labeled_folders):
        raise ValueError("You must specify an existing path for labeled folders")

    if not os.path.exists(args.folder_to_classify):
        raise ValueError("The path for label inference does not exist!")

    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    trainer = Train(args.labeled_folders, args.folder_to_classify, args.model_folder)

    print("\n*********Start training*********\n")
    trainer.train()

    print("\n*********Training done!*********\n")

    print("\n*********Start classification*********\n")
    trainer.classify()

    print("\n*********Classification done!*********\n")

else:
    if not os.path.exists(args.folder_to_classify):
        raise ValueError("The path for label inference does not exist!")

    if not os.path.exists(args.model_folder):
        raise ValueError("The path for the model does not exists!")

    elif os.path.exists(args.model_folder) and len(os.listdir(args.model_folder)) == 0:
        raise ValueError("The path for the model is empty!")

    else:
        model = Train(input_folder=None, output_folder=args.folder_to_classify, model_folder=args.model_folder)

        print("\n*********Start classification*********\n")
        model.classify()
        print("\n*********Classification done!*********\n")


############################
# Alternative with input() #
############################
# task = input("Type T for training, C for classification, TC for both")
#
# if task == 'T':
#     input_folder = input("Please, provide the path with labeled images: ")
#     output_folder = None
#     model_folder = input("Please, provide the path where the model will be saved")
#
# elif task == 'C':
#     input_folder = None
#     output_folder = input("Please, provide the path with unlabeled images: ")
#     model_folder = input("Please, provide the path where the model is saved")
#
# elif task == 'TC':
#     input_folder = input("Please, provide the path with labeled images: ")
#     output_folder = input("Please, provide the path with unlabeled images: ")
#     model_folder = input("Please, provide the path where the model will be saved")
#
# else:
#     raise ValueError("Incorrect task")
#
# model = Train(input_folder, output_folder, model_folder)
#
# if task == 'T' or task == 'TC':
#     print("\n*********Start training*********\n")
#     model.train()
#     print("\n*********Training done!*********\n")
#
# elif task == "C" or task == "TC":
#     print("\n*********Start classification*********\n")
#     model.classify()
#     print("\n*********Classification done!*********\n")
#
#
