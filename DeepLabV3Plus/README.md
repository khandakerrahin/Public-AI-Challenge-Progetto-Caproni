## DeepLabV3Plus

1. For fine-tuning the model on the damage dataset:
```
main.py
```
2. For evaluating the fine-tuned model's performance on Caproni dataset:
```
validation.py
```

Structure of the folders:

```
| caproni_topredict
    |--- FC_12_00620.jpg
    |--- FC_12_00640.jpg
    |--- ...
| damage_dataset_splitted
    | train
        |--- image
               |--- 000000000019.png
               |--- 000000000057.png
               |--- ...
        |--- mask
               |--- 000000000019.png
               |--- 000000000057.png
               |--- ...
    | test
        |--- image
               |--- 000000000001.png
               |--- 000000000016.png
               |--- ...
        |--- mask
               |--- 000000000001.png
               |--- 000000000016.png
               |--- ...
| main.py
| custom_dataset.py
| model.py
| utils.py
| validation.py
```
