import json
import urllib.request
import urllib.parse


baseurl = None


def connectServer(url):
    global baseurl
    baseurl = url


def launchGraph(graph, nframes):
    assert baseurl, 'Please connect to server first'

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
    assert baseurl, 'Please connect to server first'

    url = baseurl + '/getDescriptors'
    response = urllib.request.urlopen(url, timeout=5)
    result = response.read().decode()
    descs = json.loads(result)
    return descs
