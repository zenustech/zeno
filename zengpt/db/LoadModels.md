To load model from a file, you may use differnt nodes corresponding to different file types:
```python
# Please choose which to use according to your file extension name
model = LoadOBJModel(path='<your path>.obj')
model = LoadABCModel(path='<your path>.abc')
model = LoadFBXModel(path='<your path>.fbx')
```
Now that we support unified load with LoadModel node, regardless the file type:
```python
model = LoadModel(path='<your path>.{obj,abc,fbx}')
```
