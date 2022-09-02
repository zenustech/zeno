import ze

prim = ze.get_input('obj0').asPrim()
for i in range(len(prim.verts.pos)):
    p = prim.verts.pos[i]
    p = (p[0] + 1.0, p[1], p[2])
    prim.verts.pos[i] = p
ze.set_output('obj0', prim.asObject())
