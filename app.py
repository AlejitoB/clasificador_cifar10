from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Cargar modelo una sola vez al iniciar
model = tf.keras.models.load_model('mejor_modelo_fase2.keras')

CLASS_NAMES = ['Avión','Auto','Pájaro','Gato','Ciervo',
               'Perro','Rana','Caballo','Barco','Camión']

# Media y std de CIFAR-10 (fijos, no necesita recalcular)
MEAN = np.array([125.307, 122.950, 113.865])
STD  = np.array([62.993,  62.089,  66.705])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    file = request.files['imagen']
    img = Image.open(file.stream).convert('RGB')
    img_32 = img.resize((32, 32))
    img_array = np.array(img_32, dtype=np.float32)
    img_norm = (img_array - MEAN) / (STD + 1e-7)

    probs = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0]
    clase = int(np.argmax(probs))

    # Convertir imagen a base64 para mostrarla
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
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
