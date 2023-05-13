to simulate rigid destruction, you need to first pre-fragmentize the model to destruct into pieces:
```python
model = LoadModel(path='<your model path>.obj')
fragments = RigidPreFragmentize(model=model)
```
Then use a rigid solver to solve fraguments shattered and tangling around:
```python
rigidSolver = RigidSolver()
for frag in fragments:
    rigidSolver.addRigid(object=frag)
nframes = 100  # simulate for 100 frames
for frame in range(1, nframes + 1):
    objList = rigidSolver.getRigidObjects()
    mergedObj = MergeObjects(objects=objList)
    ExportModel(path=f'/tmp/{frame:06d}.obj', model=mergedObj)  # Export from /tmp/000001.obj to /tmp/000100.obj
    rigidSolver.step(dt=0.01)
```

