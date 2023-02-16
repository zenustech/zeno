import threading

res = []


def test1():
    return 1


def run1():
    threading.Timer(5, run1).start()
    r = test1()
    if r == 1:
        global res
        res.append(r)


def run2():
    threading.Timer(5, run2).start()
    global res
    print("--", res)


run1()
run2()
