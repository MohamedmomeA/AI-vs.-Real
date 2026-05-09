import torch

from utils.transforms import transform
from utils.model import load_model, device

model = load_model()

classes = ["AI Generated", "Real"]

def predict(image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        logits = model(image)

        probs = torch.softmax(logits, dim=1)[0]

    return {
        classes[0]: float(probs[0]),
        classes[1]: float(probs[1])
    }