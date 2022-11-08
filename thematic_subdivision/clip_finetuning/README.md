# CLIP fine tuning

#### For training run:
    python3 clip_train.py

#### For inference run:
    python3 clip_test.py


#### Structure
    clip_train.py: performs CLIP fine-tuning and save best model

    clip_test.py: performs inference on unlabeled data

    data: contains train, validation, and test image folders



Required structure of the folders (to make it work as it is):


    | clip_train.py
    | clip_test.py
    | utils.py
    | data
        |--- train
                |--- airplane
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- building
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- ...
        |--- validation
                |--- airplane
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- building
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- ...            
        |--- test
                |--- test
                        |--- image1
                        |--- image2
                        |--- ...
    

