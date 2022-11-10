# Image content extraction

#### Train and extract image content using ViT pre-trained model
####
###

#### Structure
    content_extraction_train.py: performs ViT fine-tuning for multilabel classification and save model

    content_extraction_test.py: performs test on unseen test data and returns accuracy

#### Required input:
    CSV file containing a column of image path (img_path), and a column containing the contents (content).
    Contents are N labels separated bt a comma - e.g. car, person, plant 

        