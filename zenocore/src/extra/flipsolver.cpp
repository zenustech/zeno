#include <rapidjson/document.h>
#include <filesystem>
#include <zeno/zeno.h>
#include <zeno/core/data.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/geo/geometryutil.h>
#include <zeno/utils/log.h>
#include <zeno/utils/uuid.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/flipsolver.h>
#include "glm/gtc/matrix_transform.hpp"


#undef GetObject    //winapi居然把rapidjson的GetObject方法给宏定义了


namespace zeno {

    using namespace zeno::reflect;

    DWORD WINAPI bridgeBetweenUIandSolver(LPVOID lpParam);

    #define DEBUG_NODE

    FlipSolver::FlipSolver() {
#if defined(_WIN32)
        initHandles();
#else
#endif
    }

    FlipSolver::~FlipSolver() {
        //TODO：如果进程没退出，需要发送终止信号
        terminate_solve();
        clear_cache();
    }

    NodeType FlipSolver::type() const {
        return Node_Solver;
    }

    void FlipSolver::terminate_solve() {
        //发送到线程，让线程统一做与solver进程之间的通讯
        //PostThreadMessage(m_hIPCThread, SOLVER_STOP, 0, 0); //消息的派发有一个明显的延迟
        //DWORD wtf = WaitForSingleObject(m_hSyncEvent, 5000);

        BOOL bSucceed = SetEvent(m_hSyncEvent);
        if (!bSucceed) {
            WaitForSingleObject(m_hSyncEvent, INFINITE);
        }
        //wtf = WaitForSingleObject(m_hSyncEvent, 5000);
        //clear_cache();
    }

    void FlipSolver::terminate_job() {
        CloseHandle(m_hJob);
        m_hJob = INVALID_HANDLE_VALUE;
    }

    void FlipSolver::objs_cleaned() {
        clear_cache();
    }

    void FlipSolver::CloseIPCHandles() {
        CloseHandle(m_shm_initFluid);  //初始流体共享内存文件句柄
        CloseHandle(m_shm_staticCollider);
        CloseHandle(m_hPipe_solver_read);
        CloseHandle(m_hPipe_main_write);
        CloseHandle(m_hPipe_main_read);
        CloseHandle(m_hPipe_sovler_write);  //如果solver进程被退出，这里会阻塞
        m_shm_initFluid = INVALID_HANDLE_VALUE;
        m_shm_staticCollider = INVALID_HANDLE_VALUE;
        m_hPipe_solver_read = INVALID_HANDLE_VALUE;
        m_hPipe_main_write = INVALID_HANDLE_VALUE;
        m_hPipe_main_read = INVALID_HANDLE_VALUE;
        m_hPipe_sovler_write = INVALID_HANDLE_VALUE;
    }

    void FlipSolver::initHandles() {
#if defined(_WIN32)
        CloseHandle(m_shm_initFluid);  //初始流体共享内存文件句柄
        CloseHandle(m_shm_staticCollider);
        CloseHandle(m_hPipe_solver_read);
        CloseHandle(m_hPipe_main_write);
        CloseHandle(m_hPipe_main_read);
        CloseHandle(m_hPipe_sovler_write);  //如果solver进程被退出，这里会阻塞
        m_shm_initFluid = INVALID_HANDLE_VALUE;
        m_shm_staticCollider = INVALID_HANDLE_VALUE;
        m_hPipe_solver_read = INVALID_HANDLE_VALUE;
        m_hPipe_main_write = INVALID_HANDLE_VALUE;
        m_hPipe_main_read = INVALID_HANDLE_VALUE;
        m_hPipe_sovler_write = INVALID_HANDLE_VALUE;
        m_hIPCThread = 0;

        m_hSyncEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
#else
#endif
    }

    zany FlipSolver::checkCache(std::string cache_path, int frame) {
        //checkResult的作用是：无论解算器处在什么样的状态，检查当前全局的时间帧，如果cache_path有缓存且合法的cache，
        //就作为当前节点的输出。
        std::vector<zany> objs = zeno::fromZenCache(cache_path, frame);
        if (objs.size() > 0) {
            zany output = (*objs.begin())->clone();
            return output;
        }

        //case1: 有可能这是第一次运行，cache_path已经缓存了zencache，但由于重新打开工程要标脏，所以
        //即便上一次缓存了，参数一致，也得全部删掉（至于交给谁清理后续讨论）

        //case2: 已经开始运行，但只是运行了一部分


        //case3: 已经运行结束了（solver进程已经退出）,此时路径下有cache

        return nullptr;
    }

    std::string FlipSolver::get_cachepath() const {
        const ParamPrimitive& param = m_pAdapter->get_input_prim_param("Cache Path");
        std::string cachepath;
        if (param.result.has_value()) {
            cachepath = zeno::any_cast_to_string(param.result);
        }
        else {
            cachepath = zeno::any_cast_to_string(param.defl);
        }
        return cachepath;
    }

    void FlipSolver::clear_cache() const {
        const std::string& path = get_cachepath();
        std::filesystem::path dirToRemove = std::filesystem::u8path(path);
        if (std::filesystem::exists(dirToRemove) && path.find(".") == std::string::npos)
        {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                std::error_code ec; // 防止抛异常
                std::filesystem::remove_all(entry, ec);
                if (ec) {
                    zeno::log_error("cannot remove {}, error: {}", entry.path(), ec.message());
                    return;
                }
            }
            auto& sess = zeno::getSession();
            unsigned long mainThreadId = sess.mainThreadId();
            PostThreadMessage(mainThreadId, SOLVER_CLEANUP, 0, 0);
        }
    }

    void FlipSolver::on_dirty_changed(bool bOn, DirtyReason reason, bool bWholeSubnet, bool bRecursively) {
        if (reason != Dirty_FrameChanged) {
            //清空所有cache.
            clear_cache();
        }
    }

    void FlipSolver::apply() {
        auto init_fluid = get_input_Geometry("Initialize Fluid");
        auto static_collider = get_input_Geometry("Static Collider");
        zany emission_source = clone_input("Emission Source");
        float accuracy = get_input2_float("Accuracy");
        float timestep = get_input2_float("Timestep");
        float max_substep = get_input2_float("Max Substep");
        zeno::vec3f gravity = toVec3f(get_input2_vec3f("Gravity"));
        zeno::vec3f emission = toVec3f(get_input2_vec3f("Emission Velocity"));
        bool is_emission = get_input2_bool("Is Emission");
        float dynamic_collide_strength = get_input2_float("Dynamic Collide Strength");
        float density = get_input2_float("Density");
        float surface_tension = get_input2_float("Surface Tension");
        float viscosity = get_input2_float("Viscosity");
        float wall_viscosity = get_input2_float("Wall Viscosity");
        float wall_viscosityRange = get_input2_float("Wall Viscosity Range");
        int curve_endframe = get_input2_int("Curve Endframe");
        float curve_range = get_input2_float("Curve Range");
        float preview_size = get_input2_float("Preview Size");
        float preview_minVelocity = get_input2_float("Preview Minimum Velocity");
        float preview_maxVelocity = get_input2_float("Preview Maximum Velocity");
        std::string cache_path = zsString2Std(get_input2_string("Cache Path"));
        int start_frame = get_input2_int("Begin Frame");
        int end_frame = get_input2_int("End Frame");

        if (!init_fluid || !static_collider) {
            throw makeError<UnimplError>("need init fluid and static collider");
        }
        if (cache_path.empty()) {
            throw makeError<UnimplError>("need to specify the path to store zencache");
        }
        //检测cache路径是否合理，如果不合理，直接退出
        int frame = zeno::getSession().globalState->getFrameId();
        auto spRes = checkCache(cache_path, frame);
        if (!spRes)
        {
            //在cache检测不到缓存的解算，而且solver进程不存在，此时就认为需要启动solver进程去解算，有三种可能的情况：
            //1.从来没有运行解算器
            //2.之前运行了一次，后续标脏了，因为参数类型的标脏，必须要把cache全部删掉（暂时不考虑备份，为了程序方便）
            //3.只是跑了一部分的帧，其他范围的帧要重新跑一次
#if defined(_WIN32)

            //检测Job是否存在
            if (m_hJob == INVALID_HANDLE_VALUE) {
                SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
                // 创建管道
                if (!CreatePipe(&m_hPipe_main_read, &m_hPipe_sovler_write, &sa, 0) ||
                    !CreatePipe(&m_hPipe_solver_read, &m_hPipe_main_write, &sa, 0)) {
                    //throw
                    zeno::log_error("CreatePipe failed");
                    set_output("Output", nullptr);
                    return;
                }

                m_hReadPipeEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

                std::vector<char> buf_initfluid, buf_static_collider;
                bool bConvert = encodeObject(init_fluid->toPrimitiveObject().get(), buf_initfluid);
                if (!bConvert) {
                    throw makeError<UnimplError>("init fluid encoding failed");
                }
                bConvert = encodeObject(static_collider->toPrimitiveObject().get(), buf_static_collider);
                if (!bConvert) {
                    throw makeError<UnimplError>("static collider encoding failed");
                }

                auto shm_name = generateUUID();
                auto shm_initname = shm_name + "/" + "init_fluid";
                const int shm_initfluid_size = buf_initfluid.size();
                m_shm_initFluid = CreateFileMapping(
                    INVALID_HANDLE_VALUE,
                    NULL,
                    PAGE_READWRITE,
                    0, shm_initfluid_size,
                    shm_initname.c_str()
                );
                if (m_shm_initFluid == NULL) {
                    zeno::log_error("CreateFileMapping for init fluid failed, error: {}", GetLastError());
                    set_output("Output", nullptr);
                    return;
                }

                const size_t shm_staticcoll_size = buf_static_collider.size();
                auto shm_staiccoll_name = shm_name + "/" + "static_coll";
                m_shm_staticCollider = CreateFileMapping(
                    INVALID_HANDLE_VALUE,
                    NULL,
                    PAGE_READWRITE,
                    0, shm_staticcoll_size,
                    shm_staiccoll_name.c_str()
                );
                if (m_shm_staticCollider == NULL) {
                    zeno::log_error("CreateFileMapping for static collider failed, error: {}", GetLastError());
                    set_output("Output", nullptr);
                    return;
                }

                LPVOID pInitBuf = MapViewOfFile(m_shm_initFluid, FILE_MAP_ALL_ACCESS, 0, 0, shm_initfluid_size);
                if (!pInitBuf) {
                    zeno::log_error("MapViewOfFile failed, error: {}", GetLastError());
                    CloseHandle(m_shm_initFluid);
                }
                //写入
                memcpy(pInitBuf, buf_initfluid.data(), shm_initfluid_size);

                LPVOID pStaticCollBuf = MapViewOfFile(m_shm_staticCollider, FILE_MAP_ALL_ACCESS, 0, 0, shm_staticcoll_size);
                if (!pStaticCollBuf) {
                    zeno::log_error("MapViewOfFile failed, error: {}", GetLastError());
                    CloseHandle(m_shm_staticCollider);
                }
                //写入
                memcpy(pStaticCollBuf, buf_static_collider.data(), shm_staticcoll_size);

                //创建进程和作业
                auto cmdargs = zeno::format("C:/zensolver/Debug/bin/zensolver.exe --pipe-write {} --pipe-read {} --cache-path \"{}\" --start-frame {} --end-frame {}  --init-fluid \"{}\" --size-init-fluid \"{}\" --static-collider \"{}\" --size-static-collider \"{}\"",
                    (unsigned long long)m_hPipe_sovler_write, (unsigned long long)m_hPipe_solver_read, cache_path, start_frame, end_frame, shm_initname, shm_initfluid_size, shm_staiccoll_name, shm_staticcoll_size);

                zeno::getSession().set_solver(this->m_pAdapter->get_uuid_path());

                m_hJob = CreateJobObject(NULL, NULL);
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION jobInfo = {};
                jobInfo.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(m_hJob, JobObjectExtendedLimitInformation, &jobInfo, sizeof(jobInfo));

                //配置子进程 I/O 句柄
                STARTUPINFO si = { sizeof(STARTUPINFO) };
                /* solver进程有大量的标准输出作为调试日志，故不采用标准输出作为两个进程间的管道
                si.dwFlags = STARTF_USESTDHANDLES;
                si.hStdInput = m_hPipeInRead;
                si.hStdOutput = m_hPipeOutWrite;
                si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
                */
                PROCESS_INFORMATION pi;
                if (CreateProcess(
                    (LPSTR)"C:/zensolver/Debug/bin/zensolver.exe",
                    (LPSTR)cmdargs.c_str(),
                    NULL,
                    NULL,
                    TRUE,
                    CREATE_SUSPENDED | DETACHED_PROCESS,
                    NULL,
                    NULL,
                    &si,
                    &pi))
                {
                    AssignProcessToJobObject(m_hJob, pi.hProcess);
                    ResumeThread(pi.hThread);
                }
                else {
                    zeno::log_error("CreateProcess failed");
                    set_output("Output", nullptr);
                    return;
                }

                //从解算进程读取发来的通知
                CreateThread(NULL, 0, bridgeBetweenUIandSolver, this, 0, &m_hIPCThread);
                m_hReadPipeEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
            }
            else {
                //解算进程已经在运行，但目前当前帧的cache还没结算完，只能返回空（不过返回空，也会使得程序清除脏位）
                //可以等到子进程回传消息时，再标脏位，从而重新apply
                //A: 如果用户主动杀进程，而这个句柄还在，会发生什么？

                return;
            }
#else
                //TODO: Linux下IPC
#endif

        }
        else {
            //也有一种可能，就是有cache，但解算器只是跑了一部分帧，剩下的需要用户主动触发时间轴更新，然后重新跑
            set_output("Output", std::move(spRes));
            return;
        }
        set_output("Output", nullptr);
    }

    CustomUI FlipSolver::export_customui() const {
        CustomUI ui = INode::export_customui();
        ui.uistyle.background = "#246283";
        ui.uistyle.iconResPath = ":/icons/node/fluid.svg";
        return ui;
    }

    HANDLE FlipSolver::get_pipe_read() const {
        return m_hPipe_main_read;
    }

    HANDLE FlipSolver::get_pipe_write() const {
        return m_hPipe_main_write;
    }

    HANDLE FlipSolver::get_sync_event() const {
        return m_hSyncEvent;
    }

    HANDLE FlipSolver::get_readpipe_event() const {
        return m_hReadPipeEvent;
    }

    ZENO_CUSTOMUI_NODE(FlipSolver,
        zeno::ObjectParams{
             zeno::ParamObject("Initialize Fluid", gParamType_Geometry),
             zeno::ParamObject("Static Collider", gParamType_Geometry),
             zeno::ParamObject("Emission Source", gParamType_IObject),
        },
        zeno::CustomUIParams{
            ParamTab {
                "Solver",
                {
                    ParamGroup {
                        "Solver",
                        PrimitiveParams {
                            ParamPrimitive("Accuracy", gParamType_Float, 0.8),
                            ParamPrimitive("Max Substep", gParamType_Float, 1.0),
                            ParamPrimitive("Timestep", gParamType_Float, 0.04),
                            ParamPrimitive("Gravity", gParamType_Vec3f, zeno::vec3f(0,-9.8,0)),
                            ParamPrimitive("Emission Velocity", gParamType_Vec3f, zeno::vec3f(0,-9.8,0)),
                            ParamPrimitive("Is Emission", gParamType_Bool, true),
                            ParamPrimitive("Dynamic Collide Strength", gParamType_Float, 1),
                            ParamPrimitive("Density", gParamType_Float, 1000),
                            ParamPrimitive("Surface Tension", gParamType_Float, 0),
                            ParamPrimitive("Viscosity", gParamType_Float, 0),
                            ParamPrimitive("Wall Viscosity", gParamType_Float, 0),
                            ParamPrimitive("Wall Viscosity Range", gParamType_Float, 0),
                            ParamPrimitive("Curve Endframe", gParamType_Int, 0),
                            ParamPrimitive("Curve Range", gParamType_Float, 1.1),
                            ParamPrimitive("Preview Size", gParamType_Float, 0),
                            ParamPrimitive("Preview Minimum Velocity", gParamType_Float, 0),
                            ParamPrimitive("Preview Maximum Velocity", gParamType_Float, 0),
                        }
                    },
                },
            },
            ParamTab {
                "Settings",
                {
                    ParamGroup {
                        "Settings",
                        PrimitiveParams {
                            ParamPrimitive("Begin Frame", gParamType_Int, 0),
                            ParamPrimitive("End Frame", gParamType_Int, 100),
                            ParamPrimitive("Cache Path", gParamType_String, "", ReadPathEdit),
                            ParamPrimitive("Items", gParamType_String, "Item 1", Combobox, std::vector<std::string>{"Item1", "Item2", "Item3"})
                        }
                    }
                }
            }
        },
        /*output prims:*/
        zeno::PrimitiveParams{},
        /*output objects:*/
        zeno::ObjectParams{
            zeno::ParamObject("Output", gParamType_IObject)
        },
        NodeUIStyle{ "", "" },
        "create",     //category
        "",     //nickname
        ""      //doc
    );

    static DWORD WINAPI bridgeBetweenUIandSolver(LPVOID lpParam) {
        auto& sess = zeno::getSession();
        unsigned long mainThreadId = sess.mainThreadId();

        FlipSolver* pSolver = (FlipSolver*)lpParam;

        HANDLE hPipe_main_read = pSolver->get_pipe_read();
        HANDLE hPipe_main_write = pSolver->get_pipe_write();
        HANDLE hEvent = pSolver->get_sync_event();

        //OVERLAPPED ov = {};
        //ov.hEvent = pSolver->get_readpipe_event();

        while (true) {
            OVERLAPPED ov = {};
            ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

            //发起异步读取
            DWORD bytesRead;
            char buf[1024];
            BOOL ok = ReadFile(hPipe_main_read, buf, sizeof(buf), &bytesRead, &ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING) {
                // 处理错误，通常是管道断开或句柄失效
                break;
            }

            HANDLE waitEvents[] = { hEvent, ov.hEvent };
            DWORD result = WaitForMultipleObjects(2, waitEvents, FALSE, INFINITE);

            if (result == WAIT_OBJECT_0 + 1) {
                // 读取完成
                GetOverlappedResult(hPipe_main_read, &ov, &bytesRead, FALSE);
                rapidjson::Document doc;
                doc.Parse(buf, bytesRead);
                if (!doc.IsObject()) {
                    continue;
                }
                if (!doc.HasMember("action")) {
                    continue;
                }
                const rapidjson::Value& val = doc["action"];
                std::string act = val.GetString();
                if (act == "startSolve") {
                    assert(doc.HasMember("start_frame") && doc.HasMember("end_frame"));
                    int start_frame = doc["start_frame"].GetInt();
                    int end_frame = doc["end_frame"].GetInt();
                    PostThreadMessage(mainThreadId, SOVLER_START, (WPARAM)start_frame, (LPARAM)end_frame);
                }
                else if (act == "finishFrame") {
                    if (doc.HasMember("key")) {
                        int frameComplete = doc["key"].GetInt();
                        PostThreadMessage(mainThreadId, SOLVER_UPDATE_FRAME, (WPARAM)frameComplete, (LPARAM)frameComplete);
                        zeno::log_info("frame {} from solver completes", frameComplete);
                    }
                }
                else {
                    //failed.
                }
            }
            else if (result == WAIT_OBJECT_0) {
                ResetEvent(hEvent);
                pSolver->CloseIPCHandles();
                pSolver->terminate_job();
                PostThreadMessage(mainThreadId, SOLVER_STOPPED, 0, 0);
                //MSG msg;
                //while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                //    TranslateMessage(&msg);
                //    DispatchMessage(&msg);

                //    if (msg.message == SOLVER_STOP) {
                //        //直接terminate，solver进程并不需要什么清理操作。
                //        pSolver->CloseIPCHandles();
                //        pSolver->terminate_job();
                //        PostThreadMessage(mainThreadId, SOLVER_STOPPED, 0, 0);
                //    }
                //}
            }
        }
        return 0;
    }

}