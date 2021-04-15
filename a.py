import zen
zen.loadLibrary('build/FastFLIP/libFLIPlib.so')

def execute(frame):
        import zen
        if frame == 0: zen.addNode('EndFrame', 'endFrame')
        if frame == 0: zen.addNode('RandomParticles', 'No1')
        zen.setNodeParam('No1', 'count', int(1024))
        zen.applyNode('No1')
        if frame == 0: zen.addNode('ViewParticles', 'No7')
        zen.setNodeInput('No7', 'pars', 'No1::pars')
        zen.applyNode('No7')
        zen.applyNode('endFrame')

for frame in range(1):
        print('[Zen] executing frame', frame)
        execute(frame)
