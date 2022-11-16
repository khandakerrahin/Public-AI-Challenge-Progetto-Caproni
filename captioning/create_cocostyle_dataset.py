import pandas as pd
import json

#  path to the csv file
df_dir = '/media/link92/E/ViT-GPT2/caproni.csv'
df = pd.read_csv(df_dir)[['img_path', 'img_name', 'caption']]

#  path for the results
json_store = '/media/link92/E/ViT-GPT2/dataset/dataset/annotations'


def create_json(df, task='train', out_folder=json_store):
    red_df = df[df['img_path'].str.contains(task)]
    diz = {'info': {'description': 'Custom Dataset',
                    'url': None,
                    'version': '1.0',
                    'year': 2022,
                    'contributor': 'Team11',
                    'date_created': '2022/11/08'},
           'licenses': None,
           'images': [],
           'annotations': []}

    idx = 0
    for row in red_df.iterrows():
        el_img = {
            'file_name': row[1].img_name,
            'file_path': row[1].img_path,
            'id': int(''.join([x for x in row[1].img_name if x.isnumeric()]))
        }
        el_ann = {
            'image_id': int(''.join([x for x in row[1].img_name if x.isnumeric()])),
            'id': int(idx),
            'caption': row[1].caption,
            'img_path': row[1].img_path
        }

        diz['images'].append(el_img)
        diz['annotations'].append(el_ann)
        idx += 1

    with open(f'{json_store}/annotations_{task}.json', 'w') as fp:
        json.dump(diz, fp, indent=4)
    print(f'{task} annotations done!')


#  Apply the function for each task
for task in ['train', 'test', 'validation']:
    create_json(df, task)
