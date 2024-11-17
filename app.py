import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import torch
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define directories
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'static/segmented'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

# Load YOLO model
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Save original uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # Perform tumor segmentation using YOLO model
        segmented_filename = f"segmented_{filename}"
        segmented_path = os.path.join(SEGMENTED_FOLDER, segmented_filename)

        # Run inference with YOLO model
        results = model(upload_path)
        result = results[0]
        
        # Process the segmentation mask
        image = cv2.imread(upload_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        masks = result.masks.data if result.masks else None
        mask_combined = np.zeros_like(image[:, :, 0])  # Blank mask

        if masks is not None:
            for mask in masks:
                mask = mask.cpu().numpy()
                mask_combined += mask.astype(np.uint8)

        # Overlay the mask on the original image
        plt.imshow(image)
        plt.imshow(mask_combined, alpha=0.5, cmap='jet')  # Overlay mask with transparency
        plt.axis('off')
        # Save the result as an image without using plt.show() to avoid main loop issue
        plt.savefig(segmented_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to avoid any resource issues

        # Generate URLs for the uploaded and segmented images
        uploaded_image_url = url_for('uploaded_file', filename=filename)
        segmented_image_url = url_for('segmented_file', filename=segmented_filename)
        
        return render_template('result.html', 
                               uploaded_image_url=uploaded_image_url,
                               segmented_image_url=segmented_image_url)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Route to serve segmented files
@app.route('/static/segmented/<filename>')
def segmented_file(filename):
    return send_from_directory(SEGMENTED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
