from PIL import Image
import torch

from clip_text_decoder.model import ImageCaptionInferenceModel

model = ImageCaptionInferenceModel.download_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("path/to/image.jpeg")
# The beam_size argument is optional. Larger beam_size is slower, but has
# slightly higher accuracy. Recommend using beam_size <= 3.
caption = model(image, beam_size=1)

model = ImageCaptionInferenceModel.download_pretrained("/home/a/DS/challenge/model.pt")

