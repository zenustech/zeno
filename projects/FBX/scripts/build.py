import os
import sys
import oss2
import json
import shlex
import shutil
import psutil
import signal
import codecs
import datetime
import subprocess
from flask import Flask, request

# os.name
#  nt - Windows
#  posix - Linux

PRINT_CONSOLE = False

MISSING_COPY = [
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cudart64_12.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cufft64_11.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cufftw64_11.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvrtc64_120_0.dll",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvrtc-builtins64_120.dll",
    "C:/Windows/System32/libomp140.x86_64.dll"
] if os.name == "nt" else []

BUILD_TARGET = "zenoedit.exe" if os.name == "nt" else "zenoedit"

BUILD_DST = "O:/Zeno/LL/Release" if os.name == "nt" else "/mnt/data/Zeno/LL/Release"

ZENO_SOURCE = "C:/src/zeno" if os.name == "nt" else "/opt/src/zeno"

CONFIG_FILE = "C:/src/config.json" if os.name == "nt" else "/opt/src/config.json"

VCPKG_CMAKE = "C:/DEV_PROJECT/vcpkg/scripts/buildsystems/vcpkg.cmake" if os.name == "nt" else "/home/zenus/work/vcpkg/scripts/buildsystems/vcpkg.cmake"

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
    '-DZENO_WITH_Python:BOOL=ON ' \
    '-DZENO_WITH_SampleModel:BOOL=ON ' \
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
           "<h3>{check}</h3>" \
        .format(update="update: Update zeno source",
                clean="clean: Clean build directory",
                start="start: Generate project and start building",
                build="build: Build only",
                stop="stop: Stop current task",
                restart="restart: Stop then start",
                copy="copy: Copy build to NAS:Zeno/LL/Release",
                cmake="cmake: Generate cmake project",
                check="Check Status"
                )


@app.route('/cmake')
def cmake():
    if process is None:
        run_cmd(CMAKE_OPTIONS)
        return "<h1>Generated</h1>"
    else:
        return "<h1>CMake Process Running</h1>"


@app.route('/check')
def check():
    c = int(request.args.get("mode", 1))
    if c == 0:
        check_oss()
        return "<h1>Checked OSS</h1>"
    if c == 1:
        sta = check_repo()
        return "{}".format(sta)


@app.route('/update')
def update():
    update_cmd()
    return "<h1>Updated</h1>"


@app.route('/copy')
def copy():
    if copy_cmd():
        return "<h1>Copied</h1>"
    else:
        return "<h1>Copied With Errors</h1>"


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
        update_repo = int(request.args.get("update", 1))
        print("Start: Update", update_repo)
        if update_repo == 1:
            update_cmd()
        if run_cmd(CMAKE_OPTIONS):
            if run_cmd(BUILD_COMMAND):
                if copy_cmd():
                    return '<h1>Start Build Finished</h1>'
                else:
                    return '<h1>Start Build Finished With Errors</h1>'
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
        update_repo = request.args.get("update", 1)
        if update_repo == 1:
            update_cmd()
        if run_cmd(BUILD_COMMAND):
            if copy_cmd():
                return '<h1>Build Finished</h1>'
            else:
                return '<h1>Build Finished With Errors</h1>'
        else:
            return '<h1>Build Failed</h1>'
    else:
        return "<h1> Building</h1>" \
               "{}".format(get_log())


@app.route('/restart')
def restart():
    stop_cmd()
    if request.args.get("update", 1) == 1:
        update_cmd()
    if run_cmd(CMAKE_OPTIONS):
        if run_cmd(BUILD_COMMAND):
            if copy_cmd():
                return "<h1>Restart Build Finished</h1>"
            else:
                return "<h1>Restart Build Finished With Errors</h1>"
        else:
            return "<h1>Restart Build Failed</h1>"
    else:
        return "<h1>Restart CMake Failed</h1>"


@app.route('/stop')
def stop():
    global process
    if process is not None:
        stop_cmd()
        return "<h1>Stopped</h1>"
    else:
        return "<h1>Not Started</h1>"


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

    if cmd == BUILD_COMMAND:
        target_path = "{zeno_source}/build/bin/{build_target}".format(zeno_source=ZENO_SOURCE, build_target=BUILD_TARGET)
        print("Remove Old Build Target", target_path, "Exists", os.path.exists(target_path))
        if os.path.exists(target_path):
            os.remove(target_path)

    global process

    success = False
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)

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
    sub_dir_name = "zeno-{time_str}-{os_name}".format(time_str=time_str, os_name=os.name)
    src = "{zeno_source}/build/bin".format(zeno_source=ZENO_SOURCE)
    dst = "{build_dst}/{sub_dir}".format(build_dst=BUILD_DST, sub_dir=sub_dir_name)
    pak = "{zeno_source}/build/{sub_dir}.zip".format(zeno_source=ZENO_SOURCE, sub_dir=sub_dir_name)
    target_path = "{}/{}".format(src, BUILD_TARGET)

    if os.path.exists(target_path):
        print("Copy src: ", src, " --> dst: ", dst, "exixts", os.path.exists(BUILD_DST))
        for fc in MISSING_COPY:
            shutil.copy2(fc, src)
        if os.path.exists(BUILD_DST):
            shutil.copytree(src, dst)
        
        # -j = junk-paths -y = keep links  -r = dir
        pak_command = "7z a {dst} {src}".format(dst=pak, src=src) if os.name == "nt" else "zip -rjy {dst} {src}".format(dst=pak, src=src)
        
        print("Pack src: ", src, " --> dst: ", pak, " - Command: ", pak_command)
        os.system(pak_command)
        upload_cmd(sub_dir_name+".zip", pak)
        os.remove(pak)

        return True
    else:
        print("ERROR: The build target", target_path, "doesn't exixts")
        return False


def get_oss():
    if os.path.exists(CONFIG_FILE):
        f = open(CONFIG_FILE)

        config_data = json.load(f)
        access_id = config_data["access_id"]
        access_key = config_data["access_key"]
        bucket_name = config_data["bucket_name"]

        print("OSS Config access_id", access_id)
        print("OSS Config access_key", access_key)
        print("OSS Config bucket_name", bucket_name)

        oss = ConnectOss(access_id, access_key, bucket_name)

        return oss
    else:
        print("ERROR: The config file {} doesn't exists".format(CONFIG_FILE))
        return None


def check_oss():
    oss = get_oss()
    bucket_list = oss.get_bucket_list()
    print("========== Check ==========")
    print("Oss BucketList", bucket_list)
    print("Oss Files", oss.get_all_file("download/daily-build"))
    print("===========================")


def check_repo():
    ru = os.popen("git -C {zeno_source} remote update".format(zeno_source=ZENO_SOURCE)).read().strip()
    gs = os.popen("git -C {zeno_source} status".format(zeno_source=ZENO_SOURCE)).read().strip()
    key1 = "Your branch is up to date"

    if gs.count(key1) > 0:
        return 0
    else:
        print("Remote Update:", ru)
        print("Repo Status:", gs)
        print("Found Key:", gs.count(key1))
        return 1


def upload_cmd(name, path):
    oss = get_oss()
    remote_path = "download/daily-build/win/{}".format(name) if os.name == "nt" else "download/daily-build/linux/{}".format(name)
    print("Upload Oss, RemotePath ", remote_path)
    print("Upload Oss, LocalPath ", path)
    oss.upload_file(remote_path, path)


class ConnectOss(object):
    def __init__(self, access_id, access_key, bucket_name):
        self.auth = oss2.Auth(access_id, access_key)
        self.endpoint = 'https://oss-cn-shenzhen.aliyuncs.com'
        self.bucket = oss2.Bucket(self.auth, self.endpoint, bucket_name=bucket_name)

    def get_bucket_list(self):
        """list all bucket_name under current endpoint"""
        service = oss2.Service(self.auth, self.endpoint)
        bucket_list = [b.name for b in oss2.BucketIterator(service)]
        return bucket_list

    def get_all_file(self, prefix):
        """get all file by specific prefix"""
        files = []
        for i in oss2.ObjectIterator(self.bucket, prefix=prefix):
            print("file", i.key)
            files.append(i.key)
        return files

    def read_file(self, path):
        try:
            file_info = self.bucket.get_object(path).read()
            return file_info
        except Exception as e:
            print(e, 'File does not exists')

    def download_file(self, path, save_path):
        result = self.bucket.get_object_to_file(path, save_path)
        if result.status == 200:
            print('Download Complete')

    def upload_file(self, path, local_path):
        result = self.bucket.put_object_from_file(path, local_path)
        if result.status == 200:
            print('Upload Success')


def main():
    app.run(host="0.0.0.0")


if __name__ == '__main__':
    print("------ Zeno Build -----")
    print("log file ", LOG_FILE)
    print("build command ", BUILD_COMMAND)
    print("-----------------------")
    main()
