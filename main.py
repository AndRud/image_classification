import numpy as np
import pickle
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow

app = Flask(__name__, static_folder = 'static' )

upload_folder = 'static/img/'
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route('/')
def main():
	filelist = [f for f in os.listdir(upload_folder)]
	for f in filelist:
		os.remove(upload_folder + f)
	return render_template('index.html')

def cutter(img):
    width, height = img.size
    if width == height:
        return img.resize((150,150), Image.ANTIALIAS)
    if width > height:
        left = round((width - height)/2)
        top = 0
        right = height + left
        bottom = height 
        return img.crop((left, top, right, bottom)).resize((150,150), Image.ANTIALIAS)   
    else:
        left = 0
        top = round((height - width)/2)
        right = width
        bottom = width + top 
        return img.crop((left, top, right, bottom)).resize((150,150), Image.ANTIALIAS)    

dictionary = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

@app.route('/result', methods = ['POST'])
def predict():
	file = request.files['file']
	filename = secure_filename(file.filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	if file.filename.split('.')[1] == 'jpg':
		link = 'static/img/' + file.filename
		img = Image.open(link)
		cutter_img = cutter(img)
		cutter_img.save(link)
		arr = np.asarray(cutter_img, dtype='uint8').reshape(1,150,150,3)/255
		predict_model = pickle.load(open('CV_model.sav','rb'))
		predict = predict_model.predict(arr).argmax(axis = 1)[0]
		return render_template('result.html', link = link, predict = dictionary[predict])
	else:
		return render_template('result.html')
	
	

if __name__ == '__main__':
	app.run(debug = True)
	