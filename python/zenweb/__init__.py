from flask import Flask, request

import zencli
import json


app = Flask(__name__)


@app.route('/launch', methods=['POST'])
def launch():
    graph = request.form['graph']
    nframes = request.form['nframes']

    graph = json.loads(graph)
    t = zencli.launchGraph(graph, nframes)

    return {'status': 'OK'}
