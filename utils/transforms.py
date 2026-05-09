from torchvision import transforms
from torchvision.transforms import v2

transform = transforms.Compose([
    v2.Lambda(lambda img: img.convert("RGB")),

    transforms.Resize(236),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])