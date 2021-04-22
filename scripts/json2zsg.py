#!/usr/bin/env python

import argparse
import json

from zenedit.launcher import ZenLauncher

launcher = ZenLauncher()
descs = launcher.getDescriptors()



def convert(in_data):
    in_nodes, in_links = in_data

    nodes = {}
    outputSockets = {}
    links = {}

    def parseParamValue(type, valstr):
        if type == 'string':
            return str(valstr)
        elif type == 'int':
            return int(valstr)
        elif type == 'float':
            return float(valstr)
        elif type == 'int3':
            return list(map(int, valstr.split()))
        elif type == 'float3':
            return list(map(float, valstr.split()))
        else:
            assert False, (type, valstr)

    for type, name, id, inputs, outputs, params, posx, posy in in_nodes:
        for i, sid in enumerate(outputs):
            outputSocketName = descs[type].outputs[i]
            outputSockets[sid] = name, outputSocketName

    for eid, src, dst in in_links:
        links[dst] = src

    for type, name, id, inputs, outputs, params, posx, posy in in_nodes:
        inputSockets = {}
        for i, sid in enumerate(inputs):
            inputSocketName = descs[type].inputs[i]
            if sid in links:
                inputSockets[inputSocketName] = outputSockets[links[sid]]

        parameters = {}
        for i, valstr in enumerate(params):
            paramType, paramName, _ = descs[type].params[i]
            value = parseParamValue(paramType, valstr)
            parameters[paramName] = value

        uiPosition = posx * 1.2, posy * 1.2

        nodes[name] = {
            'name': type,
            'inputs': inputSockets,
            'params': parameters,
            'uipos': uiPosition,
        }

    return nodes



ap = argparse.ArgumentParser()
ap.add_argument('infile')
ap.add_argument('outfile')
ap = ap.parse_args()

with open(ap.infile, 'r') as f:
    indata = json.load(f)

outdata = convert(indata)

with open(ap.outfile, 'w') as f:
    json.dump(outdata, f)
