from django.shortcuts import render
from django.http import JsonResponse
import json
import base64
import io
import re
from tensorflow.keras.models import load_model
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from .classifier import preprocess


# Create your views here.
model = load_model('e70lr-4binary.h5')

@csrf_exempt
def home(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        message = json.loads(data)
        name = message['name']
        response = {
            'greeting' : 'Hello ' + name + '!'
        }
        return JsonResponse(response)
    else:
        return render(request, "catNdog/index.html")

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = request.body
        message = json.loads(data)
        img_enc = message['image']
        img_enc = re.sub('^data:image/.+;base64,', '',img_enc)
        img_dec  = base64.b64decode(img_enc)
        img = Image.open(io.BytesIO(img_dec))
        img_process = preprocess(img, target_size=(250, 250))

        prediction = model.predict(img_process).tolist()
        if prediction[0][0] == 1.0:
            answer = 'DOG'
        else:
            answer = 'CAT'
        print(prediction)
        response = {
            'prediction': 'Its a ' + answer + ' !'
        }
        return JsonResponse(response)
    else:
        return render(request, "catNdog/predict.html")
    '''if request.method == 'POST':
        data = request.body.decode('utf-8')
        print(data)
        message = json.loads(data)
        print(message)
        response = {
            'prediction': {
                'dog': 1,
                'cat': 0
            }
        }
        return JsonResponse(response)
    else:
        return render(request, "catNdog/predict.html")'''


