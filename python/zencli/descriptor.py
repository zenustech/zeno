from collections import namedtuple


def parse_descriptor_line(line):
    z_name, rest = line.strip().split(':', maxsplit=1)
    assert rest.startswith('(') and rest.endswith(')'), (n_name, rest)
    inputs, outputs, params, categories = rest.strip('()').split(')(')

    z_inputs = [name for name in inputs.split(',') if name]
    z_outputs = [name for name in outputs.split(',') if name]
    z_categories = [name for name in categories.split(',') if name]

    z_params = []
    for param in params.split(','):
        if not param:
            continue
        type, name, defl = param.split(':')
        z_params.append((type, name, defl))

    return z_name, z_inputs, z_outputs, z_params, z_categories


class Descriptor(namedtuple('Descriptor',
    'inputs, outputs, params, categories')):
    pass
