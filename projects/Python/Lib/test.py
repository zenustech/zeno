# type: ignore
import ze

func0 = ze.args.obj0.asFunc()
func0()

def func(**kwargs):
    print('func called', kwargs)
    r = func0()
    print(r)
    return 42

ze.rets.obj0 = ze.ZenoObject.fromFunc(func)
