USE_WEB = 1


import zenwebapi as zenapi
#import zenapi

import zenwebvis as zenvis
#import zenvis

svraddr = None
httpurl = None
rtmpurl = None
wsurl = None

def connectServer(addr):
    global svraddr, httpurl, rtmpurl, wsurl
    svraddr = addr
    httpurl = f'http://{addr}:8000'
    rtmpurl = f'rtmp://{addr}:8001'
    wsurl = f'ws://{addr}:8000'
