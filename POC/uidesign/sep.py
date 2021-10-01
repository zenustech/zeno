with open('main.cpp') as f:
    lines = f.readlines()
    while len(lines):
        line = lines.pop(0)
        line = line.strip('\n')
        if line.startswith('struct ') and line.endswith(' {'):
            line = line[len('struct '):-len(' {')]
            print(line)
