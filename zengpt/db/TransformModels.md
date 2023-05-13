Sometimes you may need to move models to a different position, rotate the model, or scale them. We call them transforms. You may transform a model using the Transform node.
```python
model = LoadModel(path='<your path>.obj')
model = Transform(in=model, translate=(0, 0, 0), rotateQuanternion=(0, 0, 0, 1), scale=(1, 1, 1))
```
The transform is applied in this order: scale -> rotateQuanternion -> translate.
