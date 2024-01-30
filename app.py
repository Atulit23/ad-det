import requests
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gradio as gr

model = YOLO('best (4).pt')

def index(img_url):
    response = requests.get(img_url, stream=True)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    print(img_url)

    classes_ = {0: 'noti', 1: 'pop'}

    results = model.predict(source=img, conf = 0.7)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    print(boxes)
    print(classes)
    print(names)
    print(confidences)

    result_dict = {"boxes": boxes, "classes": classes, "names": names, "confidence": confidences}

    return len(boxes)


inputs_image_url = [
    gr.Textbox(type="text", label="Image URL"),
]

outputs_result_dict = [
    gr.Textbox(type="text", label="Result Dictionary"),
]

interface_image_url = gr.Interface(
    fn=index,
    inputs=inputs_image_url,
    outputs=outputs_result_dict,
    title="Popup detection",
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image_url],
    tab_names=['Image inference']
).queue().launch()