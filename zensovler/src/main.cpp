#include <QApplication>
#include "zstartup.h"
#include "zeno/utils/log.h"
#include "zeno/zeno.h"
#include "zeno/extra/EventCallbacks.h"
#include <zenoio/reader/zsgreader.h>
#include "zenosolver.h"
#include <iostream>
#include <zeno/funcs/ObjectCodec.h>
#include <thread>


static bool getBytesFromSharedMemory(const QString& shm_name, int sz, std::vector<char>& bytes)
{
    HANDLE hMapFile = OpenFileMapping(
        FILE_MAP_READ,
        FALSE,
        shm_name.toLatin1()
    );
    if (hMapFile == NULL) {
        std::cerr << "OpenFileMapping failed, error: " << GetLastError() << std::endl;
        return false;
    }
    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, sz);
    if (!pBuf) {
        std::cerr << "MapViewOfFile failed, error: " << GetLastError() << std::endl;
        return false;
    }
    bytes.resize(sz);
    memcpy(bytes.data(), pBuf, sz);
    return true;
}

void DisableStdoutBuffering() {
    //setvbuf(stdout, NULL, _IONBF, 0);  // 关闭 stdout 缓冲
    //setvbuf(stderr, NULL, _IONBF, 0);  // 关闭 stderr 缓冲
    std::cout.setstate(std::ios_base::failbit);

    //fclose(stdout);
    //fclose(stderr);
    //SetStdHandle(STD_OUTPUT_HANDLE, NULL);
    //SetStdHandle(STD_ERROR_HANDLE, NULL);
}

void AttachToConsole() {
    // 让子进程附加到自己的新控制台
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
        AllocConsole();  // 如果无法附加，则创建新控制台
    }

    // 重新打开 stdout 以确保可以输出
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
    freopen("CONIN$", "r", stdin);
}

void ReadPipeThread(HANDLE hReadPipe) {
    char buffer[1024];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
        //通知内核中止计算
        buffer[bytesRead] = '\0';
        //std::cout << "B.exe output: " << buffer;
    }
}


int main(int argc, char *argv[]) 
{
    QApplication app(argc, argv);

    MessageBoxA(0, "solver", "zensolver.exe", MB_OK);

    //AttachToConsole();
    //DisableStdoutBuffering();

#if 0
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (hStdOut == INVALID_HANDLE_VALUE || hStdOut == NULL) {
        std::cerr << "stdout handle is invalid!" << std::endl;
    }
    else if (GetConsoleMode(hStdOut, &mode)) {
        std::cout << "stdout is a console" << std::endl;
    }
    else {
        std::cout << "stdout is NOT a console, it might be redirected" << std::endl;
    }

    printf("hello world\n");
    std::cout << "std::cout helloworld" << std::endl;
    zeno::log_critical("log helloworld");
#endif

    startUp(false);

    ZENO_RECORD_RUN_INITPARAM param;

    QCommandLineParser cmdParser;
    cmdParser.addHelpOption();
    cmdParser.addOptions({
        /*第一个是名字，第二个是描述，第三个是：该选项的值名称，仅在选项需要参数时使用（如 --file <filename>*/
        {"pipe-write", "the pipe which this process write to the parent process", "0"},
        {"pipe-read", "the pipe which reads the content from parent process", "0"},
        {"init-fluid", "the name of sharedmemory that stores initialize fluid", "initialize fluid"},
        {"size-init-fluid", "the size of initialize fluid bytes", "size of sharedmemory of init-fluid"},
        {"static-collider", "the name of sharedmemory that stores static collider", "static collider"},
        {"size-static-collider", "the size of static collider", "size of sharedmemory of static-collider"},
        /*{"emission-source", "emission source", ""},
        {"size-emission-source", "the size of emission source", ""},
        {"accuracy", "", "", "0.08"},
        {"max-substep", "", "", "1"},
        {"gravity-x", "", "", "0"},
        {"gravity-y", "", "", "-9.8f"},
        {"gravity-z", "", "", "0"},
        {"emitssion-x", "", "", "0"},
        {"emitssion-y", "", "", "0"},
        {"emitssion-z", "", "", "0"},
        {"is-emission", "", "", "0"},
        {"dynamic-collide-strength", "", "", "1"},
        {"density", "", "", "1000"},
        {"surface-tension", "", "", "0"},
        {"viscosity", "", "", "0"},
        {"wall-viscosity", "", "", "0"},
        {"wall-viscosityRange", "", "", "0"},
        {"curve-force", "", "", "0"},
        {"size-curve-force", "", "", "0"},
        {"curve-endframe", "", "", "100"},
        {"curve-range", "", "", "1.1"},
        {"preview-size", "", "", "0"},
        {"preview-min-velocity", "", "", "0"},
        {"preview-max-velocity", "", "", "2"},
        {"FSD", "", "", "0"},
        {"size-FSD", "", "", "0"},*/
        });
    cmdParser.process(app);

    std::vector<char> init_fluid;
    int size_init_fluid = 0;
    std::vector<char> static_collider;
    int size_static_collider = 0;
    std::vector<char> emission_source;   /*发射源*/
    int size_emission = 0;
    float accuracy = 0.08;     /*精度*/
    float timestep = 0.04;     /*时间步长*/
    float max_substep = 1;     /*最大子步数*/
    /*重力*/
    float gravity_x = 0.f;
    float gravity_y = -9.8f;
    float gravity_z = 0.f;
    /*发射源速度*/
    float emit_vx = 0.f;
    float emit_vy = 0.f;
    float emit_vz = 0.f;
    bool is_emission = true;                            /*是否发射*/
    float dynamic_collide_strength = 1.f;                 /*动态碰撞强度*/
    float density = 1000;                          /*密度*/
    float surface_tension = 0;       /*表面张力*/
    float viscosity = 0;            /*粘性*/
    float wall_viscosity = 0;        /*壁面粘性*/
    float wall_viscosityRange = 0;   /*壁面粘性作用范围*/
    char* curve_force = 0;           /*曲线力*/
    int n_curve_force = 0;
    int curve_endframe = 100;          /*曲线终止帧*/
    float curve_range = 1.1f;           /*曲线作用范围*/
    float preview_size = 0;          /*预览大小*/
    float preview_minVelocity = 0;   /*预览最小速度*/
    float preview_maxVelocity = 2.f;   /*预览最大速度*/
    char* FSD = 0;                  /*流固耦合*/
    int n_size_FSD = 0;

    /*匿名管道句柄*/
    if (!cmdParser.isSet("pipe-write") || !cmdParser.isSet("pipe-read")) {
        zeno::log_error("init-fluid not initialize");
        return -1;
    }
    HANDLE hpipewrite = (HANDLE)cmdParser.value("pipe-write").toInt();
    HANDLE hpiperead = (HANDLE)cmdParser.value("pipe-read").toInt();

    /*初始流体*/
    if (!cmdParser.isSet("init-fluid") || !cmdParser.isSet("size-init-fluid")) {
        zeno::log_error("init-fluid not initialize");
        return -1;
    }
    QString init_fluid_shm_name = cmdParser.value("init-fluid");
    size_init_fluid = cmdParser.value("size-init-fluid").toInt();
    bool ret = getBytesFromSharedMemory(init_fluid_shm_name, size_init_fluid, init_fluid);
    if (!ret) {
        zeno::log_error("init-fluid data error");
        return -1;
    }
    std::shared_ptr<zeno::IObject> spWtf = zeno::decodeObject(init_fluid.data(), size_init_fluid);

    /*静态碰撞体*/
    if (!cmdParser.isSet("static-collider") || !cmdParser.isSet("size-static-collider")) {
        zeno::log_error("static-collider not initialize");
        return -1;
    }
    QString static_collider_shm_name = cmdParser.value("static-collider");
    size_static_collider = cmdParser.value("size-static-collider").toInt();
    ret = getBytesFromSharedMemory(static_collider_shm_name, size_static_collider, static_collider);
    if (!ret) {
        zeno::log_error("init-fluid data error");
        return -1;
    }
    std::shared_ptr<zeno::IObject> spWtf2 = zeno::decodeObject(static_collider.data(), size_static_collider);

    if (cmdParser.isSet("emission-source") && cmdParser.isSet("size-emission-source")) {
        //emission_source = cmdParser.value("emission-source").toStdString().data();
        //size_emission = cmdParser.value("size-emission-source").toInt();
    }

    if (cmdParser.isSet("accuracy")) {
        accuracy = cmdParser.value("accuracy").toFloat();
    }
    if (cmdParser.isSet("max-substep")) {
        max_substep = cmdParser.value("max-substep").toFloat();
    }
    if (cmdParser.isSet("gravity-x")) {
        gravity_x = cmdParser.value("gravity-x").toFloat();
    }
    if (cmdParser.isSet("gravity-y")) {
        gravity_y = cmdParser.value("gravity-y").toFloat();
    }
    if (cmdParser.isSet("gravity-z")) {
        gravity_z = cmdParser.value("gravity-z").toFloat();
    }
    if (cmdParser.isSet("emitssion-x")) {
        emit_vx = cmdParser.value("emitssion-x").toFloat();
    }
    if (cmdParser.isSet("emitssion-y")) {
        emit_vy = cmdParser.value("emitssion-y").toFloat();
    }
    if (cmdParser.isSet("emitssion-z")) {
        emit_vz = cmdParser.value("emitssion-z").toFloat();
    }
    if (cmdParser.isSet("is-emission")) {
        is_emission = cmdParser.value("is-emission").toInt();
    }
    if (cmdParser.isSet("dynamic-collide-strength")) {
        dynamic_collide_strength = cmdParser.value("dynamic-collide-strength").toFloat();
    }
    if (cmdParser.isSet("density")) {
        density = cmdParser.value("density").toFloat();
    }
    if (cmdParser.isSet("surface-tension")) {
        surface_tension = cmdParser.value("surface-tension").toFloat();
    }
    if (cmdParser.isSet("viscosity")) {
        viscosity = cmdParser.value("viscosity").toFloat();
    }
    if (cmdParser.isSet("wall-viscosity")) {
        wall_viscosity = cmdParser.value("wall-viscosity").toFloat();
    }
    if (cmdParser.isSet("wall-viscosityRange")) {
        wall_viscosityRange = cmdParser.value("wall-viscosityRange").toFloat();
    }
    if (cmdParser.isSet("curve-force") && cmdParser.isSet("size-curve-force")) {
        curve_force = cmdParser.value("curve-force").toStdString().data();
        n_curve_force = cmdParser.value("size-curve-force").toInt();
    }
    if (cmdParser.isSet("curve-endframe")) {
        curve_endframe = cmdParser.value("curve-endframe").toInt();
    }
    if (cmdParser.isSet("curve-range")) {
        curve_range = cmdParser.value("curve-range").toFloat();
    }
    if (cmdParser.isSet("preview-size")) {
        preview_size = cmdParser.value("preview-size").toFloat();
    }
    if (cmdParser.isSet("preview-min-velocity")) {
        preview_minVelocity = cmdParser.value("preview-min-velocity").toFloat();
    }
    if (cmdParser.isSet("preview-max-velocity")) {
        preview_maxVelocity = cmdParser.value("preview-max-velocity").toFloat();
    }
    if (cmdParser.isSet("FSD") && cmdParser.isSet("size-FSD")) {
        FSD = cmdParser.value("FSD").toStdString().data();
        n_size_FSD = cmdParser.value("size-FSD").toInt();
    }

    std::string s_init_fluid_shm_name = init_fluid_shm_name.toStdString();
    std::string s_static_collider_shm_name = static_collider_shm_name.toStdString();

    //std::cerr.rdbuf(std::cout.rdbuf());
    //std::clog.rdbuf(std::cout.rdbuf());
    //zeno::set_log_stream(std::clog);

    //启动一个监听线程，如果主进程主动通知sovler进程关闭，则让解算程序自行结束
    std::thread thdListenMainProc(ReadPipeThread, hpiperead);
    thdListenMainProc.detach();

    zensolver::flip_solve(
        hpipewrite,
        s_init_fluid_shm_name,      /*初始流体*/
        size_init_fluid,
        s_static_collider_shm_name,   /*静态碰撞体*/
        size_static_collider,
        "",   /*TODO: 发射源*/
        0,
        accuracy,     /*精度*/
        timestep,     /*时间步长*/
        max_substep,     /*最大子步数*/
        /*重力*/
        gravity_x,
        gravity_y,
        gravity_z,
        /*发射源速度*/
        emit_vx,
        emit_vy,
        emit_vz,
        is_emission,                            /*是否发射*/
        dynamic_collide_strength,               /*动态碰撞强度*/
        density,               /*密度*/
        surface_tension,       /*表面张力*/
        viscosity,             /*粘性*/
        wall_viscosity,        /*壁面粘性*/
        wall_viscosityRange,   /*壁面粘性作用范围*/
        curve_force,           /*曲线力*/
        n_curve_force,
        curve_endframe,        /*曲线终止帧*/
        curve_range,           /*曲线作用范围*/
        preview_size,          /*预览大小*/
        preview_minVelocity,   /*预览最小速度*/
        preview_maxVelocity,   /*预览最大速度*/
        FSD,                   /*流固耦合*/
        n_size_FSD
    );

    return app.exec();
}
