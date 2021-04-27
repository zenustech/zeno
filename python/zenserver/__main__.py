from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from . import app


host, port = '0.0.0.0', 8000
print('Listening at', 'http://' + host + ':' + str(port))
server = pywsgi.WSGIServer((host, port), app, handler_class=WebSocketHandler)
server.serve_forever()
