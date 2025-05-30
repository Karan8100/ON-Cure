from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

model = load_model('models/my_model.keras')

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    predicted_class = class_labels[predicted_class_index]

    if predicted_class == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor Type: {predicted_class.capitalize()}", confidence_score

# New route for upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, confidence = predict_tumor(filepath)

            return render_template('upload.html',
                                   result=result,
                                   confidence=f"{confidence*100:.2f}%",
                                   file_path=f'/uploads/{file.filename}')
    # GET request
    return render_template('upload.html', result=None)

# Other routes for your static pages
@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/protection')
def protection():
    return render_template('protection.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
