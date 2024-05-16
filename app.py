import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input


app = Flask(__name__)

# Load your trained model
model = load_model('crop_disease_vgg16_model.h5')

print('Model loaded. Check http://127.0.0.1:5000/')

# Update the labels dictionary based on your crop disease classes
labels = {0: 'Healthy', 1: 'Unhealthy', 2: 'Unhealthy'}

# Function to process the uploaded image with the model
def get_result(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size based on your model input shape
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess the input image based on your model requirements
    predictions = model.predict(x)[0]
    return predictions

# Route to render the index.html template
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle image upload and predict
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions = get_result(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
