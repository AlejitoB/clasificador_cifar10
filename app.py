from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = None

CLASS_NAMES = ['Avión','Auto','Pájaro','Gato','Ciervo',
               'Perro','Rana','Caballo','Barco','Camión']

MEAN = np.array([125.307, 122.950, 113.865])
STD  = np.array([62.993,  62.089,  66.705])

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'mejor_modelo_fase2.keras'))
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    m = get_model()
    file = request.files['imagen']
    img = Image.open(file.stream).convert('RGB')
    img_32 = img.resize((32, 32))
    img_array = np.array(img_32, dtype=np.float32)
    img_norm = (img_array - MEAN) / (STD + 1e-7)

    probs = m.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0]
    clase = int(np.argmax(probs))

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        'clase': CLASS_NAMES[clase],
        'confianza': round(float(probs[clase]) * 100, 1),
        'probabilidades': {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(10)},
        'imagen': img_b64
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
