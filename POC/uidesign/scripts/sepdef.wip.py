with open('main.cpp') as f:
    lines = f.readlines()
    while len(lines):
        line = lines.pop(0).strip('\n')
        if line.startswith('struct ') and line.endswith(' {'):
            line = line[len('struct '):-len(' {')]
            while len(lines):
                line = lines.pop(0).strip('\n')
                if line == '};':
                    break
                print('!!!', line)
        print('===========')
