from PIL import Image, ImageFilter
from glob import glob
import random
import torch
import numpy as np


class Transform:
    def __init__(self, image, deg, width, height, keep_track=False, custom_probs=None):
        self.image = image
        self.deg = deg
        self.width = width
        self.height = height
        self.keep_track = keep_track
        if custom_probs is not None:
            self.probs = custom_probs
        else:
            self.probs = [1, 1]

        self.idx = 2

    def rotate(self):
        return self.image.rotate(self.deg, Image.NEAREST, expand=1)

    def rescale(self):
        new_height = int(self.width * self.image.size[1] / self.image.size[0])
        new_width = int(self.height * self.image.size[0] / self.image.size[1])
        return self.image.resize((new_width, new_height), Image.ANTIALIAS)

    def horizontal_flip(self):
        return self.image.transpose(method=Image.Transpose.TRANSPOSE.FLIP_LEFT_RIGHT)

    def vertical_flip(self):
        return self.image.transpose(method=Image.Transpose.TRANSPOSE.FLIP_TOP_BOTTOM)

    def transpose(self):
        return self.image.transpose(method=Image.TRANSPOSE)

    def gaussian_blur(self):
        return self.image.filter(ImageFilter.GaussianBlur(radius=2))

    def smooth(self):
        return self.image.filter(ImageFilter.SMOOTH)

    def detail(self):
        return self.image.filter(ImageFilter.DETAIL)

    def edge_enhance(self):
        return self.image.filter(ImageFilter.EDGE_ENHANCE)

    def sharpen(self):
        return self.image.filter(ImageFilter.SHARPEN)

    def apply_transformations(self):
        self.image = self.image.rotate(self.deg)
        self.image = self.rescale()

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.horizontal_flip()
        else:
            if random.random() > 0.5:
                self.image = self.horizontal_flip()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.vertical_flip()
        else:
            if random.random() > 0.5:
                self.image = self.vertical_flip()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.transpose()
        else:
            if random.random() > 0.5:
                self.image = self.transpose()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.gaussian_blur()
        else:
            if random.random() > 0.5:
                self.image = self.gaussian_blur()
                self.probs.append(0)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.smooth()
        else:
            if random.random() > 0.5:
                self.image = self.smooth()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.detail()
        else:
            if random.random() > 0.5:
                self.image = self.detail()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.edge_enhance()
        else:
            if random.random() > 0.5:
                self.image = self.edge_enhance()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track is False:
            if self.probs[self.idx] == 1:
                self.image = self.sharpen()
        else:
            if random.random() > 0.5:
                self.image = self.sharpen()
                self.probs.append(1)
            else:
                self.probs.append(0)
        self.idx += 1

        if self.keep_track:
            return self.image, self.probs

        return self.image


def linear_gradient(band, start, end):
    # if just one pixel band, return it without applying gradient
    if start - end == 0:
        return band
    else:
        band_len = end - start + 1
        mid_point = band_len // 2
        # even number of pixels to treat
        if band_len % 2 == 0:
            indices = [i for i in range(mid_point-1, -1, -1)]
            indices.extend([i for i in range(0, mid_point)])
        # odd number of pixels to treat
        else:
            indices = [i for i in range(mid_point, 0, -1)]
            indices.extend([i for i in range(0, mid_point+1)])
    for i, j in zip(range(start, end+1), indices):
        band[:, i, 3] = 255 - j * 2
    return band


def colorize_mask(image, kind):
    diz = {'bands': [0],
           'stains': [1],
           'scratches': [2]}

    arr = np.array(image)

    for ch in range(len(arr.shape)):
        if ch in diz[kind]:
            arr[:, :, ch][arr[:, :, 3] != 0] = 255
        else:
            arr[:, :, ch][arr[:, :, 3] != 0] = 0

    return Image.fromarray(arr).convert('RGBA')


def choose_coord(max_size=512):
    x, y = random.randint(0, max_size), random.randint(0, max_size)
    return x, y


def create_band(max_size=512):
    white = random.randint(0, 1) == 0
    apply_lin_grad = random.randint(0, 2) > 0  # change to > 0!!!

    background = np.full((max_size, max_size), 255, dtype='uint8')
    alpha = np.full((max_size, max_size), 255, dtype='uint8')

    start, _ = choose_coord(max_size=max_size)
    end = random.randint(0, max_size)
    start, end = sorted([start, end])

    background[:, start:end+1] = 0
    alpha[background != 0] = 0

    if white:
        background[np.where(background == 0)] = 255
        alpha[np.where(background != 255)] = 0

    ret = np.dstack((background, background, background, alpha))
    mask = Image.fromarray(ret).convert('RGBA')
    if apply_lin_grad:
        ret = linear_gradient(ret, start, end)

    ret = Image.fromarray(ret).convert('RGBA')

    return ret, mask


def superimpose(image, list_of_damages, max_size=512):
    n, m = max_size, max_size
    background_mask = Image.fromarray(np.zeros((n, m, 4), dtype='uint8')).convert('RGBA')

    while list_of_damages:
        curr_dmg = list_of_damages.pop()
        curr_dmg_kind = curr_dmg.split('/')[-2]
        if curr_dmg_kind == 'bands':
            curr_dmg, msk = create_band(max_size)
            image.paste(curr_dmg, (0, 0), mask=curr_dmg)
            msk = colorize_mask(msk, curr_dmg_kind)
            background_mask.paste(msk, (0, 0), mask=msk)
        else:
            curr_dmg = Image.open(curr_dmg).convert('RGBA')
            coord_x, coord_y = choose_coord(max_size=512)
            mask = colorize_mask(curr_dmg, curr_dmg_kind)
            # Transform for all the errors except for bands

            deg = random.randint(a=-180, b=180)
            rand_width = random.uniform(a=0.1 * 512, b=1.2 * 512)
            rand_height = random.uniform(a=0.1 * 512, b=1.2 * 512)

            t_curr_dmg, probs = Transform(curr_dmg, deg, rand_width, rand_height, keep_track=True).apply_transformations()
            t_mask = Transform(mask, deg, rand_width, rand_height, custom_probs=probs).apply_transformations()
            # t_mask = colorize_mask(t_mask, curr_dmg_kind)
            image.paste(t_curr_dmg, (coord_x, coord_y), mask=t_curr_dmg)
            background_mask.paste(t_mask, (coord_x, coord_y), mask=t_mask)

    return image, background_mask


def check_order_and_fix(list_of_damages):
    res = []
    band_counter = 0
    band = 'a/bands/a'
    for dmg in list_of_damages:
        dmg_name = dmg.split('/')[-2]
        print(dmg_name)
        if dmg_name == 'bands':
            band_counter += 1
        else:
            res.append(dmg)
    res.extend([band] * band_counter)

    return res


class Damage:
    def __init__(self, damages_dir, max_dmg=6):
        self.dmg_kind = glob(f'{damages_dir}/*')
        self.max_dmg = max_dmg
        self.chosen_damages = []

    def choose_one(self):
        return random.choice(self.dmg_kind)

    def choose_multiple(self):
        n_damages = random.randint(1, self.max_dmg)
        for kind in range(n_damages):
            curr_kind = self.choose_one()
            curr_kind_name = curr_kind.split('/')[-1]
            if curr_kind_name == 'bands':
                if random.random() > 0.1:
                    while curr_kind_name == 'bands':
                        curr_kind = self.choose_one()
                        curr_kind_name = curr_kind.split('/')[-1]
                    curr_dmg = random.choice([i for i in glob(f'{curr_kind}/*')])
                else:
                    curr_dmg = 'idontexist/bands/idontexist'
            else:
                curr_dmg = random.choice([i for i in glob(f'{curr_kind}/*')])
            self.chosen_damages.append(curr_dmg)

        return self.chosen_damages


class DamageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, damages_dir, max_damage=6, new_size=512):
        self.images = glob(f'{root_dir}/*')
        self.damages_dir = damages_dir
        self.images.sort()
        self.max_damage = max_damage
        self.new_size = new_size

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).resize((self.new_size, self.new_size)).convert('LA').convert('RGBA')
        dmg_instance = Damage(self.damages_dir, max_dmg=self.max_damage)
        chosen_damages = check_order_and_fix(dmg_instance.choose_multiple())
        image, mask = superimpose(image, chosen_damages, max_size=self.new_size)
        image, mask = image.convert('L'), mask.convert('RGB')

        return image, mask

    def __len__(self, idx):
        return len(self.images)
