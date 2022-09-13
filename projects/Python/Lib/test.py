import ze

func0 = ze.args.obj0.asFunc()
func0()

def func(**kwargs):
    print('func called', kwargs)
    return 42

ze.rets.obj0 = ze.ZenoObject.fromFunc(func)
