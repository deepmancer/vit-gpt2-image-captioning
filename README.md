# Image-Captioning
## Project Description
This project aims to implement an image captioning module that uses the Google/ViT-Base-Patch16-224-In21k pretrained model as the encoder and the DistilGPT-2 pretrained model as the decoder, using the Transformers library. The goal of the project is to generate captions for images in the COCO dataset, by encoding the image features using the ViT model and decoding them using the GPT-2 model. The model is fine-tuned on the COCO dataset for image captioning task, and the performance is evaluated using Rouge, Bleu, and Bert metrics.

## Dataset
The dataset used in this project is the COCO dataset, which contains a large collection of images with corresponding captions. The dataset is split into 118,000 training examples and 5,000 validation examples, with each image having 5 corresponding captions.

## Implementation Details
The implementation of the image captioning module is done using the PyTorch deep learning framework and the Transformers library. The ViT and GPT-2 models are pretrained models that are fine-tuned on the COCO dataset for image captioning task. The ViT model is used to encode the image features and the GPT-2 model is used to decode them into a caption. The model is fine-tuned for 2 epochs.
