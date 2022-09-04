import ze

prim = ze.args['obj0'].asPrim()
for i in range(prim.verts.size()):
    print(prim.verts['pos'][i])
