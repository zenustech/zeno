import zen
zen.loadLibrary('build/libzenbase.so')
print(zen.dumpDescriptors())
# ===============
def execute(frame):
        import zen
        if frame == 0: zen.addNode('ReadParticles', 'No1')
        zen.setNodeParam('No1', 'path', str(f'../monkey.obj'))
        zen.applyNode('No1')
        if frame == 0: zen.addNode('SimpleSolver', 'No8')
        zen.setNodeInput('No8', 'ini_pars', 'No1::pars')
        zen.setNodeParam('No8', 'dt', float(0.04))
        zen.setNodeParam('No8', 'G', zen.float3(0, 0, 1))
        zen.applyNode('No8')
        if frame == 0: zen.addNode('WriteParticles', 'No4')
        zen.setNodeInput('No4', 'pars', 'No8::pars')
        zen.setNodeParam('No4', 'path', str(f'/tmp/{frame:06d}.obj'))
        zen.applyNode('No4')

for frame in range(64):
        print('[Zen] executing frame', frame)
        execute(frame)
# ===============
