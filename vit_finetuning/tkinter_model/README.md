# Tkinter GUI

### Basic GUI for training and classification using ViT pre-trained model
###
It is possible to use it for 

1) Training and classification:
   * create a folder with labeled images: it must contain a folder for each class.
     * suggested ~100 images per class.
   * provide the path for: the labeled images, unlabeled images, folder where the model will be saved.
2) Classification:
   * provide the path for: existing trained model, unlabeled images.

###
At the end of both processes, a new folder "classification_result" inside the folder of unlabeled images will be created. 
It will contain the folders of classified images.

N.B classified images are a copy of the unlabeled images.