# type: ignore
import ze

func0 = ze.args.obj0.asFunc()
func0()
func0()

def func(**kwargs):
    print('func called', kwargs)
    print(func0())
    return 42

ze.rets.obj0 = ze.ZenoObject.fromFunc(func)
