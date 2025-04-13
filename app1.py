from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import requests

app = Flask(__name__)

# OneDrive URL to your model file
model_url = "https://1drv.ms/u/c/3a7e9e95600e8051/EURViyY9BN1AllEiR1BAcAgBKwOmJMHVtG_d8yrlybw5Rw?e=VR6sRu"

# Download the model from OneDrive
def download_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Save the model file locally
        with open('VGG16_model.h5', 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully")
    else:
        print("Failed to download model")
        raise Exception("Failed to download model")

# Download the model on startup
download_model(model_url)

# Load the model after download
model = load_model('VGG16_model.h5')

# Adjust these to match your model input
IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = image.resize(IMG_SIZE).convert('RGB')
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    img = Image.open(file.stream)
    img_tensor = preprocess_image(img)
    prediction = model.predict(img_tensor)
    predicted_class_index = np.argmax(prediction[0])
    
    # Use the class_names list you provided
    class_names = ['Central Serous Chorioretinopathy', 'Diabetic Retinopathy', 'Disc Edema',
                   'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia', 'Pterygium',
                   'Retinal Detachment', 'Retinitis Pigmentosa']
    
    predicted_class = class_names[predicted_class_index]
    
    return jsonify({'prediction': predicted_class})
    
if __name__ == '__main__':
    app.run(debug=True)
