import ze

prim = ze.get_input('obj0').asPrim()
retprim = ze.ZenoPrimitiveObject.new()
retprim.verts.resize(prim.verts.size())
for i in range(prim.verts.size()):
    p = prim.verts.pos[i]
    p = (p[0] + 1.0, p[1], p[2])
    retprim.verts.pos[i] = p
ze.set_output('obj0', retprim)
