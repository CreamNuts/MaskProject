from flask import Flask, request, render_template, url_for, redirect

import sys
import os
import torch
from PIL import Image
import cv2
from inference import MyModel

# load model
model = MyModel('Edit.pt', 'Map.pt')

# load server
HOST = '0.0.0.0'
PORT = 8890
app = Flask(__name__)


# no cache
@app.after_request
def set_response_headers(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r

# catch all route to 'login'
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def index(path):
    return redirect(url_for('login'))

# login-page
@app.route('/login')
def login():
    return render_template("login.html")

# login val
@app.route('/image', methods=['POST'])
def login_success():
    # 'switch_page' returns 0 or 1, if no exists None
    if request.form.get('switch_page') != None:
        return render_template("img-inference.html")
    
    # password validation
    userfile = open('user.txt', 'r')
    user_result = userfile.read()
    if request.method == 'POST':
        user_pw = request.form['password']
    
        if user_pw == user_result:
            # print('----------- log) login success')
            return render_template("img-inference.html")
        else:
            # print('----------- log) login failed')
            return redirect(url_for('login'))

# image-inference page        
@app.route('/img-inference', methods=['POST'])
def image_predict():
    # print('----------- log) inference image')
    # remove 'result.png'
    if os.path.isfile('./static/result.png'):
        os.remove('./static/result.png')
        
    if request.method == 'POST':
        # call 'inference' again
        if request.files['image'].filename == '':
            file = request.form['image_path']
        # get image from remote
        else :
            file = request.files['image']
        
        # inference
        img = Image.open(file).convert('RGB')
        img.save('./static/input.png')

        res =  model.predict(img)
        res.save('./static/result.png')

        # return a path of 'input.png', 'result.png' 
        input_url = url_for('static', filename='input.png')
        output_url = url_for('static', filename='result.png')
            
        return render_template("img-inference.html", input_img='.'+input_url, output_img='.'+output_url)

# run server
if __name__ == '__main__':
    app.run(host=HOST, debug=True, port=PORT)
