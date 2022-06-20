import os
import sys
import subprocess

if sys.platform == 'linux':
    print('linux detected')
    subprocess.check_call([
        'wget',
        'https://zenustech.com/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.sh',
        '-O',
        'install-optix.sh',
    ])
    subprocess.check_call([
        'bash',
        'install-optix.sh',
        '--skip-license',
        '--prefix=./optix-sdk',
    ])
elif sys.platform == 'win32':
    print('windows detected')
    import requests
    r = requests.get(
        'https://zenustech.com/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.sh'
    )
    with open('install-optix.exe', 'wb') as f:
        f.write(r.content)
    subprocess.check_call([
        'install-optix.exe',
        '--skip-license',
        '--prefix=./optix-sdk',
    ])
else:
    assert False, sys.platform

