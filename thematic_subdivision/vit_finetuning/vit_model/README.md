# ViT fine tuning

##### Train and classify using ViT pre-trained model
####
It creates a new folder "classification_result" inside the "folder_to_classify". It contains labeled folders with inside the classified images (copy).
###
#### For training and classification run:
    python3 main.py --labeled_folders /labeled_path/ --folder_to_classify /unlabeled_path/ --model_folder /model_path/


#### For only classification run:
    python3 main.py --folder_to_classify /unlabeled_path/ --model_folder /model_path/

###
N.B. For training and classification, if the "model_path" does not exist it will be created
###


#### Example of folder structure:

    | labeled_path
        |--- train
                |--- airplane
                        |--- image1
                        |--- image2
                        |--- ...
                |--- building
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- ...
        |--- validation
                |--- airplane
                        |--- image1
                        |--- image2
                        |--- ...
                |--- building
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- ...            
    | unlabeled_path
        |-- image1
        |-- image2
        |-- image3
        |-- ...

    | model_folder
        