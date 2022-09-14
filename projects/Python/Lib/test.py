# type: ignore
import ze

func0 = ze.args.obj0
func0.asFunc()()
func0.asFunc()()

def func(**kwargs):
    print('func called', kwargs)
    f = func0.asFunc()
    r = f()
    print(r)
    return 42

ze.rets.obj0 = ze.ZenoObject.fromFunc(func)
