from flask import Flask, request

import zencli
import json


app = Flask(__name__)


@app.route('/launchGraph', methods=['POST'])
def launchGraph():
    graph = json.loads(request.form['graph'])
    nframes = request.form['nframes']

    t = zencli.launchGraph(graph, nframes)
    return 'OK'


@app.route('/getDescriptors', methods=['GET'])
def getDescriptors():
    descs = zencli.getDescriptors()
    return json.dumps(descs)
