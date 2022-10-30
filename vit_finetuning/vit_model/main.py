import argparse
from final_model import *


parser = argparse.ArgumentParser(description='please provide path to labeled folders and path to '
                                             'folder with unlabeled images')
parser.add_argument('--labeled_folders', type=str, required=True,
                    help='input folder with labeled data')
parser.add_argument('--folder_to_classify',  type=str, required=True,
                    help='folder with images to classify')
parser.add_argument('--model_folder',  type=str, required=True,
                    help='folder to save model checkpoints')

args = parser.parse_args()

# Alternative with input()
#
# input_folder = input("Please, provide the path with labeled images: ")
# output_folder = input("Please, provide the path with unlabeled images: ")

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
