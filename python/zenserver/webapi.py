from flask import request
import json

import zenapi
from zenutils import go

from . import app


@app.route('/launchGraph', methods=['POST'])
def launchGraph():
    graph = json.loads(request.form['graph'])
    nframes = request.form['nframes']

    t = go(zenapi.launchGraph, graph, nframes)
    return 'OK'


@app.route('/getDescriptors', methods=['GET'])
def getDescriptors():
    descs = zenapi.getDescriptors()
    return json.dumps(descs)
