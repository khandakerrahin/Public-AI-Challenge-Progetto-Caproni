# ViT fine tuning

#### For training run:
    python3 VIT_classifier.py

#### For test run:
    python3 test_vit.py


#### Structure
    vit_train.py: performs ViT fine-tuning and save model

    vit_test.py: performs test on unseen test data and returns accuracy



Required structure of the folders (to make it work as it is):

    | vit_train.py
    | vit_test.py
    | ../clip_finetuning/data
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
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- building
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- ...            
        |--- test
                |--- airplane
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- building
                        |--- imagee1
                        |--- image2
                        |--- ...
                |--- ...            

