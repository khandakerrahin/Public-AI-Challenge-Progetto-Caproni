# App

### Basic GUI for thematic subdivision, metadata extraction, and damage assessment using tkinter.
###
![alt text](./imgs/start.png)

###
For all the processes, select the input folder that contains all the images
1. In the thematic subdivision processes, a new folder "thematic_subdivision_result",  inside the folder of the input images will be created. 
It will contain the folders of classified images.

    N.B classified images are a copy of the unlabeled images.


2. In the metadata extraction, a csv file "metadata_results.csv" will be created in the provided folder. It will contain the image paths, their subject, content,
description, and damage level.


3. In the damage assessment process, a new folder "damage_assessment_result",  inside the folder of the input images will be created. 
It will contain the folders of classified images.

    N.B classified images are a copy of the unlabeled images.


##


![alt text](./imgs/tasks.png)

###

To install the requirements run
      
      pip install -r requirements.txt

To install OFA transformer model and download the checkpoint refer to
      
      https://huggingface.co/OFA-Sys/ofa-large

The other checkpoints are available here

      https://drive.google.com/drive/folders/1gtqUgeDxxEeMTkBRgL-2rIbkJRL11VAW?usp=sharing
      

To make an Ubuntu application
   * edit the folder path in demo_app.desktop

   * Then run in the terminal


      sudo cp /path_to_folder/demo_app.desktop /usr/share/applications/demo_app.desktop
