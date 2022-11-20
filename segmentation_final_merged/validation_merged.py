import PIL
import ntpath
import cv2
from model import load_model
from utils import mask_to_consider, from_segmentation_to_mask, calculate_damage_score
from custom_dataset import ValidationDataset
import argparse
import gc
import warnings

import torch.nn.functional as F
from PIL import Image, ImageFile

from detection_models import networks
from detection_util.util import *

warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


def blend_mask(img, mask):
    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def merge_masks(scr_mask, all_mask):
    all_mask = PIL.Image.fromarray(all_mask)
    amask = all_mask.getdata()
    scr_mask = scr_mask.getdata()
    new_image = []
    for scr_mask_pixel, all_mask_pixel in zip(scr_mask, amask):
        if scr_mask_pixel[0] in list(range(200, 256)):
            new_image.append(scr_mask_pixel)
        else:
            new_image.append(all_mask_pixel)

    # update image data
    all_mask.putdata(new_image)

    return all_mask


def main(config):
    # loading DeepLabV3Plus model checkpoint
    # Specific .pth model to load
    load_from = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/models/EP0008~1.PTH'

    # Images for which to predict a mask
    root_dir = 'C:/Users/shake/Downloads/Compressed/Public-AI-Challenge-Progetto-Caproni-Ludovico/Public-AI-Challenge-Progetto-Caproni-Ludovico/DeepLabV3Plus/caproni_topredict'
    # Folder where to store predicted masks
    results_folder = 'predicted'

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    classes = ['background', 'bands', 'stains', 'dots', 'scratches']
    select_class_rgb_values = mask_to_consider(classes)

    DEVICE = torch.device('cpu')

    model = load_model(DEVICE, n_classes=len(classes), load_from=load_from)

    # ValidationDataset loads only images (not masks) from the provided root_dir
    caproni_dataset = ValidationDataset(root_dir=root_dir,
                                        preprocessing=None,
                                        class_rgb_values=select_class_rgb_values)

    print("initializing the dataloader")

    model_scratch = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

    # load model for Scratch detection
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/detection/FT_Epoch_latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_scratch.load_state_dict(checkpoint["model_state"])
    print("model weights loaded for scratches")

    if config.GPU >= 0:
        model_scratch.to(config.GPU)
    else:
        model_scratch.cpu()
    model_scratch.eval()

    for idx in range(len(caproni_dataset)):
        image, img_name = caproni_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        # processing using DeepLabV3Plus model
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = from_segmentation_to_mask(pred_mask)
        image = np.transpose(image, (1, 2, 0))

        print("processing", img_name)

        # processing using Scratch detection model
        scratch_file = img_name
        if not os.path.isfile(scratch_file):
            print("Skipping non-file %s" % image)
            continue
        scratch_image = Image.open(scratch_file).convert("RGB")
        w, h = scratch_image.size

        transformed_image_PIL = data_transforms(scratch_image, config.input_size)
        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        _, _, ow, oh = scratch_image.shape
        scratch_image_scale = scale_tensor(scratch_image)

        if config.GPU >= 0:
            scratch_image_scale = scratch_image_scale.to(config.GPU)
        else:
            scratch_image_scale = scratch_image_scale.cpu()
        with torch.no_grad():
            P = torch.sigmoid(model_scratch(scratch_image_scale))

        P = P.data.cpu()
        P = F.interpolate(P, [ow, oh], mode="nearest")

        tv.utils.save_image(
            (P >= 0.4).float(),
            os.path.join(
                "checkpoints/",
                "temp" + ".png",
            ),
            nrow=1,
            padding=0,
            normalize=True,
        )
        scr_img = Image.open(os.path.join("checkpoints/", "temp" + ".png", ))
        newsize = (224, 224)
        scr_img = scr_img.resize(newsize)

        # merging the results of both model
        merged_mask = merge_masks(scr_img, pred_mask)

        # calculating total percentage of damage
        damage_score = calculate_damage_score(merged_mask)

        # TODO does not work yet, needs fixing
        # damage_score = calculate_weighted_damage_score(merged_mask)

        filename = ntpath.basename(img_name)[:-4]

        cv2.imwrite(f'{results_folder}/{filename}_{damage_score}.png', np.hstack([image, merged_mask]))

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_name', type=str, default="FT_Epoch_latest.pt", help='Checkpoint Name')

    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--test_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
    config = parser.parse_args()

    main(config)
