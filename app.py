from flask import Flask, request, jsonify, render_template
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load Azure Cognitive Services credentials from .env file
endpoint = os.getenv('AZURE_CV_ENDPOINT')
subscription_key = os.getenv('AZURE_CV_KEY')

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and call Azure OCR
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Read the image with Azure Computer Vision OCR
    with open(filepath, 'rb') as image_stream:
        ocr_result = computervision_client.read_in_stream(image_stream, raw=True)

    # Get the operation ID from the response headers
    operation_id = ocr_result.headers['Operation-Location'].split('/')[-1]

    # Retrieve the results
    while True:
        result = computervision_client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break

    # Extract text from the results
    extracted_text = ''
    if result.status == 'succeeded':
        for page in result.analyze_result.read_results:
            for line in page.lines:
                extracted_text += line.text + '\n'

    return jsonify({'text': extracted_text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
