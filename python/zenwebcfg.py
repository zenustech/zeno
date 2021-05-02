#import zenwebapi as zenapi
import zenapi

#import zenwebvis as zenvis
import zenvis

baseurl = None

def connectServer(url):
    global baseurl, zenapi, zenvis
    baseurl = url
