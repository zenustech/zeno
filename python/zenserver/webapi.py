from flask import request
import json

import zenapi

from . import app


@app.route('/launchGraph', methods=['POST'])
def launchGraph():
    graph = json.loads(request.form['graph'])
    nframes = request.form['nframes']

    t = zenapi.launchGraph(graph, nframes)
    return 'OK'


@app.route('/getDescriptors', methods=['GET'])
def getDescriptors():
    descs = zenapi.getDescriptors()
    return json.dumps(descs)
