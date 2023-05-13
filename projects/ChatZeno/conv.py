import json
import os
import shutil

injson = 'desc.jsonl'
outdir = 'refs'
shutil.rmtree(outdir, ignore_errors=True)
os.mkdir(outdir)

def process(data):
    data = json.loads(data)
    with open(f'refs/{data[0]}.md', 'w') as f:
        def p(*args, **kwargs):
            print(*args, file=f, **kwargs)
        p(f"```python")
        p(f"# {data[-1] or data[0]}")
        if len(data[2]) == 1 or len(data[2]) == 0:
            p(f"zeno.{data[0]}(")
        else:
            p(f"noderes = zeno.{data[0]}(")
        def cihou(arg):
            arg_type, arg_name, arg_default, arg_desc = arg
            def k(*xs):
                if arg_type_post == ' or None':
                    return None
                elif len(xs) == 1:
                    return xs[0]
                else:
                    return xs
            def m(f):
                def w(x):
                    if arg_type_post == ' or None':
                        return f(x) if x else None
                    else:
                        return f(x) if x else f()
                return w
            arg_type_post = ''
            if arg_name == 'SRC' or arg_name == 'DST': return
            if arg_type.startswith("optional "):
                arg_type = arg_type[len("optional "):]
                arg_type_post = ' or None'
            if arg_type == "vec3f":
                arg_value = repr(tuple(map(m(float), arg_default.split(","))) if arg_default else k(0, 0, 0))
            elif arg_type == "vec3i":
                arg_value = repr(tuple(map(m(int), arg_default.split(","))) if arg_default else k(0, 0, 0))
            elif arg_type == "vec2f":
                arg_value = repr(tuple(map(m(float), arg_default.split(","))) if arg_default else k(0, 0))
            elif arg_type == "vec2i":
                arg_value = repr(tuple(map(m(int), arg_default.split(","))) if arg_default else k(0, 0))
            elif arg_type == "vec4f":
                arg_value = repr(tuple(map(m(float), arg_default.split(","))) if arg_default else k(0, 0, 0, 0))
            elif arg_type == "vec4i":
                arg_value = repr(tuple(map(m(int), arg_default.split(","))) if arg_default else k(0, 0, 0, 0))
            elif arg_type == "float":
                arg_value = repr(m(float)(arg_default))
            elif arg_type == "int":
                arg_value = repr(m(int)(arg_default))
            elif arg_type == "bool":
                arg_value = repr(m(bool)(arg_default))
            elif arg_type in ("string", "multiline_string", "readpath", "writepath"):
                if arg_type == "readpath":
                    arg_type = "str, readable path"
                elif arg_type == "writepath":
                    arg_type = "str, writeable path"
                elif arg_type == "multiline_string":
                    arg_type = "str, may have multiple lines"
                else:
                    arg_type = "str"
                arg_value = repr(arg_default if arg_default else f'<{arg_name}>')
            elif arg_type in ("list", "ListObject"):
                arg_type = "list"
                arg_value = repr(list())
            elif arg_type in ("dict", "DictObject"):
                arg_type = "dict"
                arg_value = repr(dict())
            elif arg_type in ("prim", "primitive", "PrimitiveObject"):
                arg_type = "ze.ZenoPrimitiveObject"
                arg_value = f"<{arg_name}>"
            elif arg_type in ("numeric", "NumericObject"):
                arg_type = "any numeric types including int float vec3f vec3i"
                arg_value = str(m(float)(arg_default) if ',' in arg_default else tuple(map(m(float), arg_default.split(","))))
            elif arg_type in ("zany", "IObject"):
                arg_type = "any type"
                arg_value = f"<{arg_name}>"
            elif arg_type.startswith("enum "):
                options = arg_type[len("enum "):].split()
                if options and arg_default not in options:
                    arg_default = options[0]
                arg_value = repr(arg_default)
                arg_type = f"options are: {' '.join(options)}"
            else:
                # if arg_type != "":
                    # print(f"warning: unknown type {arg_type}")
                if arg_type == "" and not arg_type_post:
                    arg_default = repr(None)
                arg_value = arg_default or f'<{arg_name}>'
            p(f"    {arg_name}={arg_value}," + ("" if not arg_type and not arg_type_post and not arg_desc else f" # {arg_type}{arg_type_post}{', ' + arg_desc if arg_desc and (arg_type or arg_type_post) else ''}"))
        for arg in data[1] + [(x, y + '_', z, w) for x, y, z, w in data[3]]:
            cihou(arg)
        def cihoutype(arg_type):
            if arg_type in ("string", "multiline_string", "readpath", "writepath"):
                if arg_type == "readpath":
                    arg_type = "str, readable path"
                elif arg_type == "writepath":
                    arg_type = "str, writeable path"
                else:
                    arg_type = "str"
            elif arg_type in ("list", "ListObject"):
                arg_type = "list"
            elif arg_type in ("dict", "DictObject"):
                arg_type = "dict"
            elif arg_type in ("prim", "primitive", "PrimitiveObject"):
                arg_type = "ze.ZenoPrimitiveObject"
            elif arg_type in ("numeric", "NumericObject"):
                arg_type = "any numeric types including int float vec3f vec3i"
            elif arg_type in ("zany", "IObject"):
                arg_type = ""
            return arg_type
        if len(data[2]) == 0:
            p(")")
        elif len(data[2]) == 1:
            arg_type, arg_name, _, arg_desc = data[2][0]
            comment = ''
            if arg_type or arg_desc:
                arg_type = cihoutype(arg_type)
                comment = f" # {arg_type}{', ' + arg_desc if arg_desc and (arg_type) else ''}"
            p(f").{arg_name}{comment}")
        else:
            p(")")
            for arg in data[2]:
                arg_type, arg_name, _, arg_desc = arg
                if arg_name == 'SRC' or arg_name == 'DST': return
                comment = ''
                if arg_type or arg_desc:
                    arg_type = cihoutype(arg_type)
                    comment = f" # {arg_type}{', ' + arg_desc if arg_desc and (arg_type) else ''}"
                p(f"noderes.{arg_name}{comment}")
        p(f"```")

with open(injson, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            process(line)
