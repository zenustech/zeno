import os
import shutil
import sys
import shlex
import psutil
import signal
import codecs
import datetime
import subprocess
from flask import Flask

PRINT_CONSOLE = False

MISSING_COPY = [
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cudart64_12.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cufft64_11.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cufftw64_11.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvrtc64_120_0.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvrtc-builtins64_120.dll",
    "C:/Windows/System32/libomp140.x86_64.dll"
]

BUILD_DST = "O:/Zeno/LL/Release"

ZENO_SOURCE = "C:/src/zeno"

VCPKG_CMAKE = "C:/DEV_PROJECT/vcpkg/scripts/buildsystems/vcpkg.cmake"

SIMPLE_OPTIONS = \
    'cmake ' \
    '-S{zeno_source} ' \
    '-B{zeno_source}/build ' \
    '-DCMAKE_TOOLCHAIN_FILE={vcpkg_cmake} ' \
    '-GNinja ' \
    '-Wno-dev ' \
        .format(zeno_source=ZENO_SOURCE,
                vcpkg_cmake=VCPKG_CMAKE)

CMAKE_OPTIONS = \
    'cmake ' \
    '-S{zeno_source} ' \
    '-B{zeno_source}/build ' \
    '-DCMAKE_TOOLCHAIN_FILE={vcpkg_cmake} ' \
    '-GNinja ' \
    '-Wno-dev ' \
    '-DZENOFX_ENABLE_OPENVDB:BOOL=ON ' \
    '-DZENOFX_ENABLE_LBVH:BOOL=ON ' \
    '-DZENO_ENABLE_OPTIX:BOOL=ON ' \
    '-DZENO_SYSTEM_OPENVDB:BOOL=OFF ' \
    '-DZENO_MULTIPROCESS=ON ' \
    '-DCMAKE_BUILD_TYPE=Release ' \
    '-DZENO_WITH_ZenoFX:BOOL=ON ' \
    '-DZENO_WITH_zenvdb:BOOL=ON ' \
    '-DZENO_WITH_FastFLIP:BOOL=ON ' \
    '-DZENO_WITH_Rigid:BOOL=ON ' \
    '-DZENO_WITH_oldzenbase:BOOL=ON ' \
    '-DZENO_WITH_TreeSketch:BOOL=ON ' \
    '-DZENO_WITH_Functional:BOOL=ON ' \
    '-DZENO_WITH_Alembic:BOOL=ON ' \
    '-DZENO_WITH_FBX:BOOL=ON ' \
    '-DZENO_WITH_CalcGeometryUV:BOOL=ON ' \
    '-DZENO_WITH_MeshSubdiv:BOOL=ON ' \
    '-DZENO_WITH_CuLagrange:BOOL=ON ' \
    '-DZENO_WITH_CuEulerian:BOOL=ON ' \
    '-DZENO_WITH_CUDA:BOOL=ON ' \
    '-DZENO_WITH_TOOL_FLIPtools:BOOL=ON ' \
    '-DZENO_WITH_TOOL_BulletTools:BOOL=ON ' \
    '-DZENO_WITH_Python:BOOL=OFF ' \
    '-DZENO_ENABLE_OPENMP:BOOL=ON ' \
        .format(zeno_source=ZENO_SOURCE,
                vcpkg_cmake=VCPKG_CMAKE)

BUILD_COMMAND = \
    'cmake '\
    '--build {zeno_source}/build '\
    '--config Release '\
    '--target zenoedit '\
    '--parallel 32 ' \
        .format(zeno_source=ZENO_SOURCE)

LOG_FILE = \
    "{zeno_source}/build/log.txt" \
        .format(zeno_source=ZENO_SOURCE)


app = Flask(__name__)
process = None


@app.route('/help')
def allcommand():
    return "<h3>{update}</h3>" \
           "<h3>{clean}</h3>" \
           "<h3>{start}</h3>" \
           "<h3>{build}</h3>" \
           "<h3>{stop}</h3>" \
           "<h3>{restart}</h3>" \
           "<h3>{copy}</h3>" \
           "<h3>{cmake}</h3>" \
        .format(update="update: Update zeno source",
                clean="clean: Clean build directory",
                start="start: Generate project and start building",
                build="build: Build only",
                stop="stop: Stop current task",
                restart="restart: Stop then start",
                copy="copy: Copy build to NAS:Zeno/LL/Release",
                cmake="cmake: Generate cmake project"
                )


@app.route('/cmake')
def cmake():
    if process is None:
        run_cmd(CMAKE_OPTIONS)
        return "<h1>Generated</h1>"
    else:
        return "<h1>CMake Process Running</h1>"


@app.route('/update')
def update():
    update_cmd()
    return "<h1>Updated</h1>"


@app.route('/copy')
def copy():
    copy_cmd()
    return "<h1>Copied</h1>"


@app.route('/clean')
def clean():
    if process is None:
        clean_cmd()
        return "<h1>Cleaned</h1>"
    else:
        return "<h1>Clean Process Running</h1>"


@app.route('/start')
def start():
    if process is None:
        update_cmd()
        if run_cmd(CMAKE_OPTIONS):
            if run_cmd(BUILD_COMMAND):
                copy_cmd()
                return '<h1>Start Build Finished</h1>'
            else:
                return '<h1>Start Build Failed</h1>'
        else:
            return '<h1>Start CMake Failed</h1>'
    else:
        return "<h1>The build is in progress</h1>" \
               "{}".format(get_log())


@app.route('/build')
def build():
    if process is None:
        update_cmd()
        if run_cmd(BUILD_COMMAND):
            copy_cmd()
            return '<h1>Build Finished</h1>'
        else:
            return '<h1>Build Failed</h1>'
    else:
        return "<h1> Building</h1>" \
               "{}".format(get_log())


@app.route('/stop')
def stop():
    global process
    if process is not None:
        stop_cmd()
        return "<h1>Stopped</h1>"
    else:
        return "<h1>Not Started</h1>"


@app.route('/restart')
def restart():
    stop_cmd()
    update_cmd()
    if run_cmd(CMAKE_OPTIONS):
        if run_cmd(BUILD_COMMAND):
            copy_cmd()
            return "<h1>Restart Build Finished</h1>"
        else:
            return "<h1>Restart Build Failed</h1>"
    else:
        return "<h1>Restart CMake Failed</h1>"


def get_log():
    f = open(LOG_FILE, "r")
    data = f.readlines()
    f.close()
    return "<br>".join(data)


def stop_cmd():
    global process

    if process is None:
        print("Stop Process is None")
        return

    print("Kill ", process.pid)
    pid = process.pid

    try:
        if os.name == "nt":
            # os.kill(process.pid, signal.SIGTERM)
            # process.terminate()
            # process.send_signal(signal.SIGKILL)
            # os.system("taskkill /PID {} /F".format(pid))

            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                print("Kill pid ", child.pid)
                child.kill()
            parent.kill()
    except Exception as e:
        print("Stop Exception: ", e)

    process = None

    try:
        if os.path.exists(LOG_FILE):
            print("Remove log file")
            os.remove(LOG_FILE)
    except Exception as e:
        print("Stop Exception Remove ", e)


def run_cmd(cmd):
    print("Build Options ", cmd)
    global process

    success = False
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    dir_name = os.path.dirname(LOG_FILE)
    if not os.path.exists(dir_name):
        print("Create log dir ", dir_name)
        os.mkdir(dir_name)
        #open(LOG_FILE, 'a').close()

    with codecs.open(LOG_FILE, "a", "utf-8") as logfile:
        logfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        logfile.write("\n")
        logfile.write(cmd)
        logfile.write("\n")

        while True:
            if process is None:
                print("Run Process is none - readline")
                break

            line = process.stdout.readline().decode("utf-8")
            if line:
                logfile.write(line)
                if PRINT_CONSOLE:
                    sys.stdout.write(line)

            if process is None:
                print("Run Process is none - poll")
                break
            if process.poll() is not None:
                print("Run Process Poll")
                success = True
                break

    print("Run Finished")
    process = None

    return success


def update_cmd():
    os.system("git -C {zeno_source} reset --hard".format(zeno_source=ZENO_SOURCE))
    os.system("git -C {zeno_source} pull".format(zeno_source=ZENO_SOURCE))
    os.system("git -C {zeno_source} submodule update --init --recursive".format(zeno_source=ZENO_SOURCE))


def clean_cmd():
    os.system("git -C {zeno_source} clean -xfd".format(zeno_source=ZENO_SOURCE))


def copy_cmd():
    time_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M%S")
    src = "{zeno_source}/build/bin".format(zeno_source=ZENO_SOURCE)
    dst = "{build_dst}/{sub_dir}".format(build_dst=BUILD_DST, sub_dir="zeno-{time_str}".format(time_str=time_str))
    if os.path.exists(src):
        print("Copy src: ", src, " --> dst: ", dst)
        shutil.copytree(src, dst)
    for fc in MISSING_COPY:
        shutil.copy2(fc, dst)

def main():
    app.run(host="0.0.0.0")


if __name__ == '__main__':
    print("------ Zeno Build -----")
    print("log file ", LOG_FILE)
    print("build command ", BUILD_COMMAND)
    print("-----------------------")
    main()
