# type: ignore
import ze

def on_update(state):
    keys = ze.no.ZGL_StateGetKeys(state=state, type='STRING').keys.split()
    if keys:
        print('keys are', keys)

    offset = [0, 0, 0]
    dx = 0.01
    if 'A' in keys:
        offset[0] = -dx
    if 'S' in keys:
        offset[1] = -dx
    if 'W' in keys:
        offset[1] = +dx
    if 'D' in keys:
        offset[0] = +dx

    if any(offset):
        theBox = ze.no.PortalOut(name_='theBox').port
        ze.no.PrimTranslate(prim=theBox, offset=offset)

ze.rets.on_update = ze.ZenoObject.fromFunc(on_update)
