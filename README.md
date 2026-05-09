# AI vs Real Image Detector

A deep learning project for detecting AI-generated images using a fine-tuned ConvNeXt Tiny model.

## Features

- Detects whether an image is AI-generated or real
- Built with PyTorch and Gradio
- Uses ConvNeXt Tiny architecture
- Achieved 94.79% validation accuracy

## Dataset

Dataset used:
ai-generated-images-vs-real-images

### Fake Images
- Stable Diffusion
- MidJourney
- DALL-E

### Real Images
- Pexels
- Unsplash
- WikiArt

Dataset size: 52.41 GB

## Model

Architecture:
- ConvNeXt Tiny backbone with a binary classification head (2 neurons)

Frameworks:
- PyTorch
- Torchvision
- Gradio

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
'''

## Run the app

run the app

'''bash
python app.py
'''
