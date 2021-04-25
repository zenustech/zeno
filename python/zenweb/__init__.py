from flask import Flask, request

import zencli


app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    param = request.form['param']
    return f'Hello, {param}!'
