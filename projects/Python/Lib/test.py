import ze

def func():
    print('func called')
    return 42

ze.rets.obj0 = ze.ZenoObject._newFunc(func)
