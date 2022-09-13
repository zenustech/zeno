import ze

def func(**kwargs):
    print('func called', kwargs)
    return 42

ze.rets.obj0 = ze.ZenoObject.fromFunc(func)
