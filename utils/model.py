import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():

    model = torch.load(
        "model/Final_AI_Vs_Real_convnext.pth",
        map_location=device,
        weights_only=False
    )

    # لو كان محفوظ بـ DataParallel
    if hasattr(model, "module"):
        model = model.module

    model.eval()

    return model.to(device)