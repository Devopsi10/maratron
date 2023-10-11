import timm
import transformers
from torch import nn
import numpy as np
import gradio as gr
import PIL
import pandas as pd

from obs import ObsClient
import os

import uuid
from datetime import datetime

obsClient = ObsClient(
    access_key_id='RJ7XOBMDQE3IKNNU0P1X',
    secret_access_key='UO77PwR7e3TmhXSq5dfkRN2beeIvYuqNh7y0gBfI',
    server='obs.tr-west-1.myhuaweicloud.com'
)
flagged_dir = './flagged'
files_to_upload = [f for f in os.listdir(flagged_dir) if os.path.isfile(os.path.join(flagged_dir, f))]

def upload_all_flagged():
    for file in files_to_upload:
        file_path = os.path.join(flagged_dir, file)
        upload_to_obs(file_path)

def upload_to_obs(file_path):
    bucket_name = 'byss'
    object_key = os.path.basename(file_path)  # Use the filename as the object key
    obsClient.putFile(bucketName=bucket_name, objectKey=object_key, file_path=file_path)

# Call the function to upload all flagged files



# Instantiate classification model
from fastai.vision.all import *
model_multi = load_learner('vit_tiny_patch16.pkl')

def binary_label(path):
    return 'No-anomaly' if (parent_label(path) == 'No-Anomaly') else 'Anomaly'

model_binary = load_learner('vit_tiny_patch16_binary.pkl')

# Instantiate segmentation model
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torchvision.transforms import Grayscale

seg_feature_extractor = SegformerFeatureExtractor.from_pretrained('zklee98/segformer-b1-solarModuleAnomaly-v0.1')
seg_model = SegformerForSemanticSegmentation.from_pretrained('zklee98/segformer-b1-solarModuleAnomaly-v0.1')

def get_seg_overlay(image, seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(sidewalk_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    #img = PIL.Image.open(img)

    return img

#@title `def sidewalk_palette()`

def sidewalk_palette():
    """Sidewalk palette that maps each class to RGB values."""
    return [
        [0, 0, 0],
        [216, 82, 24],
        [255, 255, 0],
        [125, 46, 141],
        [118, 171, 47],
        [161, 19, 46],
        [255, 0, 0],
        [0, 128, 128],
        [190, 190, 0],
        [0, 255, 0],
        [0, 0, 255],
        [170, 0, 255],
        [84, 84, 0],
        [84, 170, 0],
        [84, 255, 0],
        [170, 84, 0],
        [170, 170, 0],
        [170, 255, 0],
        [255, 84, 0],
        [255, 170, 0],
        [255, 255, 0],
        [33, 138, 200],
        [0, 170, 127],
        [0, 255, 127],
        [84, 0, 127],
        [84, 84, 127],
        [84, 170, 127],
        [84, 255, 127],
        [170, 0, 127],
        [170, 84, 127],
        [170, 170, 127],
        [170, 255, 127],
        [255, 0, 127],
        [255, 84, 127],
        [255, 170, 127],
    ]


COST_PER_UNIT_AREA_CRACKING = 250
COST_PER_UNIT_AREA_SHADOWING = 0
COST_PER_UNIT_AREA_CELL = 187
COST_PER_UNIT_AREA_SOILING = 150
COST_PER_UNIT_AREA_NO_ANOMALY = 0
TOTAL_AREA_OF_PANEL = 100





def predict(classification_mode, image):


    if (classification_mode == 'Binary Classification'):
        model = model_binary
    else:
        model = model_multi

    labels = model.dls.vocab
    # Classification model prediction
    #image  = PILImage.create(image)
    pred, pred_idx, probs = model.predict(image)

    seg_img = None
    percentage_affected = '0%'
    if (pred.upper() != 'NO-ANOMALY'):
        addChannel = Grayscale(num_output_channels=3)
        image = addChannel(image)

        inputs = seg_feature_extractor(images=image, return_tensors="pt")
        outputs = seg_model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False)

        # Second, apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        seg_img = get_seg_overlay(image, pred_seg)

        classified_pixels = np.unique(pred_seg.numpy(), return_counts=True)
        pixels_count = dict({classified_pixels[0][0]: classified_pixels[1][0],
                             classified_pixels[0][1]: classified_pixels[1][1]})
        #percentage_affected = round((pixels_count[1]/960)*100, 1)
        percentage_affected = round((pixels_count[1]/(pixels_count[0]+pixels_count[1]))*100, 1)
        percentage_affected = str(percentage_affected) + '%'

        #seg_img = PIL.Image.fromarray(seg_img)
        affected_area_in_m2 = (pixels_count[1]/(pixels_count[0]+pixels_count[1])) * TOTAL_AREA_OF_PANEL
        anomaly_cost_map = {
            "CRACKING": COST_PER_UNIT_AREA_CRACKING,
            "SHADOWING": COST_PER_UNIT_AREA_SHADOWING,
            "CELL": COST_PER_UNIT_AREA_CELL,
            "SOILING": COST_PER_UNIT_AREA_SOILING,
            "NO-ANOMALY": COST_PER_UNIT_AREA_NO_ANOMALY
        }
        
        specific_cost = anomaly_cost_map.get(pred.upper(), 0)
        repair_cost = specific_cost * affected_area_in_m2

        if pred.upper() != 'NO-ANOMALY':
          upload_all_flagged()


    return ({labels[i]: float(probs[i]) for i in range(len(labels))}, seg_img, percentage_affected, f"${repair_cost:.2f}")





description = """
<center><img src="https://cdn.leonardo.ai/users/ea51b01d-0fc3-464f-964d-31246eb82933/generations/0d8baade-445e-414d-b17e-20c95c12adff/Default_I_have_a_projet_which_is_about_drone_usage_for_solar_p_3.jpg" width=270px> </center><br>
<center>This program identifies the type of anomaly found in solar panel using an image classification model and percentage of the affected area using an image segmentation model.</center><br><br><br>
<center> Step 1: Choose classification mode >   Step 2: Upload your image >   Step 3: Click Submit    |    Examples available below</center><br>
"""

gr.Interface(fn=predict,
             inputs= [gr.Dropdown(choices=['Binary Classification', 'Multiclass Classification'], label='(Step 1) Classification Mode:',
                                  info='Choose to classify between anomaly and no-anomaly (Binary) OR between 12 different types of anomalies (Multi).').style(container=False),
                      gr.Image(type='pil', label='(Step 2) Input infrared image: ').style(container=False)],
             outputs=[gr.outputs.Label(num_top_classes=3, label='Detected:').style(container=False),
                      gr.Image(type='pil', label=' ').style(height=240, width=144),
                      gr.Textbox(label='Affected area:').style(container=False),
                      gr.Textbox(label='Repair Cost:').style(container=False)],  # Added Repair Cost output
             title='Solar Panel Anomaly Detector',
             description=description,
             examples=[['Binary Classification', '4849.jpg'], ['Multiclass Classification', '4849.jpg'],
                       ['Binary Classification', '7016.jpg'], ['Multiclass Classification', '10000.jpg']],
             cache_examples= False,
             article= '<center>by SolInspect</center>').launch(debug=True)



