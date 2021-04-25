import json
import urllib.request
import urllib.parse


baseurl = 'http://localhost:8000'


def launchGraph(graph, nframes):
    params = {
        'graph': json.dumps(graph),
        'nframes': nframes,
    }
    data = urllib.parse.urlencode(params).encode()
    url = baseurl + '/launchGraph'
    response = urllib.request.urlopen(url, data=data, timeout=5)
    result = response.read().decode()
    assert result == 'OK', result


def getDescriptors():
    url = baseurl + '/getDescriptors'
    response = urllib.request.urlopen(url, timeout=5)
    result = response.read().decode()
    descs = json.loads(result)
    return descs
