# type: ignore
import ze

key = ze.args.arg
prim = ze.args.prim.asPrim()

dx = 0.1
offset = None
if key == 'A':
    offset = [-dx, 0, 0]
elif key == 'S':
    offset = [0, -dx, 0]
elif key == 'W':
    offset = [0, +dx, 0]
elif key == 'D':
    offset = [+dx, 0, 0]

if offset is not None:
    ze.no.PrimTranslate(prim=prim, offset=offset)

ze.rets.ret = 0
