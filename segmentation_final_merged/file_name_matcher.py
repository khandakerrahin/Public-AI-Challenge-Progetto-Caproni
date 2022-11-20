from pathlib import Path
import shutil
import os

# First update the directories and Run to remove unmatched files from the first directory
# Then Swap first and second DIR to remove unmatched files from the second directory
firstDir = Path('C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/damage_dataset_v4/train/image')
secondDir = Path('C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/damage_dataset_v4/train/mask')

difference = (set(map(lambda p: p.relative_to(firstDir), firstDir.rglob('*'))) -
              set(map(lambda p: p.relative_to(secondDir), secondDir.rglob('*'))))

print(difference)

if len(difference) > 0:
    print('\nContent to be deleted:\n')
    for a in difference:
        a2 = Path(firstDir, a)
        print('   ', a2)
    while True:
        copyornot = input('\nDelete? (Y/n):\n')
        if copyornot == 'Y':
            break
        elif copyornot == 'n':
            print('...')
            continue
        else:
            print('(Y/n)')
    for a in difference:
        a2 = Path(firstDir, a)
        if os.path.isfile(a2):
            os.remove(a2)
        if os.path.isdir(a2):
            shutil.rmtree(a2)
        print('\nDone.')