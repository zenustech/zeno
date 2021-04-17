import zen
zen.loadLibrary('build/FastFLIP/libFLIPlib.so')

def substep():
        zen.substepBegin()
        if zen.G.substepid == 0: zen.addNode('RunOnce', 'No39')
        zen.applyNode('No39')
        if zen.G.substepid == 0: zen.addNode('RandomParticles', 'No20')
        zen.setNodeInput('No20', 'COND', 'No39::cond')
        zen.setNodeParam('No20', 'count', int(1024))
        zen.applyNode('No20')
        if zen.G.substepid == 0: zen.addNode('NumericFloat', 'No13')
        zen.setNodeParam('No13', 'value', float(0.1))
        zen.applyNode('No13')
        if zen.G.substepid == 0: zen.addNode('IntegrateFrameTime', 'No1')
        zen.setNodeInput('No1', 'desired_dt', 'No13::value')
        zen.applyNode('No1')
        if zen.G.substepid == 0: zen.addNode('RunAfterFrame', 'No32')
        zen.applyNode('No32')
        if zen.G.substepid == 0: zen.addNode('ViewParticles', 'No26')
        zen.setNodeInput('No26', 'pars', 'No20::pars')
        zen.setNodeInput('No26', 'SRC', 'No1::DST')
        zen.setNodeInput('No26', 'COND', 'No32::cond')
        zen.applyNode('No26')
        zen.substepEnd()

def execute():
        zen.frameBegin()
        while zen.substepShouldContinue():
                substep()
        zen.frameEnd()

for frame in range(1):
        print('[Zen] executing frame', frame)
        execute()
