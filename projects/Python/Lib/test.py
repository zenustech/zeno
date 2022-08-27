import ze

graph = ze.ZenoGraph()
msg = ze.ZenoObject.makeString('hello python')
print(graph.callTempNode('Route', {'input': msg}))
