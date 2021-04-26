from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from . import app


host, port = '0.0.0.0', 8000
server = pywsgi.WSGIServer((host, port), app, handler_class=WebSocketHandler)
server.serve_forever()
