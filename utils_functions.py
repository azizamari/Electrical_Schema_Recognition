from PIL import Image
import torch
import io


def get_image_from_bytes(binary_image, max_size=1024):
    input_image =Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((
        int(input_image.width * resize_factor),
        int(input_image.height * resize_factor)
    ))
    return resized_image

def get_yolov5():
    model = torch.hub.load('./yolov5', 'custom', path='./model/modelv1.pt', source='local')
    model.conf = 0.5
    return model