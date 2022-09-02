import ze

prim = ze.get_input('obj0').asPrim()
print(prim.verts['pos'][0])
print(prim.verts.keys())
print('pos' in prim.verts)
