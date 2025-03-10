#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <zeno/utils/log.h>
#include <zeno/utils/Timer.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/GraphException.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/zeno.h>
#include <string>
#include <zeno/utils/scope_exit.h>
#include "corelaunch.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include "zstartup.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>
#include <zenomodel/include/jsonhelper.h>


static FILE *ourfp;
static char ourbuf[1 << 20]; // 1MB

struct Header { // sync with viewdecode.cpp
    size_t total_size;
    size_t info_size;
    size_t magicnum;
    size_t checksum;

    void makeValid() {
        magicnum = 314159265;
        checksum = total_size ^ info_size ^ magicnum;
    }
};

static void send_packet(HANDLE hPipeWrite, std::string_view info) {
    DWORD nBytesWritten;
    WriteFile(hPipeWrite, info.data(), info.length(), &nBytesWritten, 0);
}

int runner_start(std::string const &progJson, int sessionid, HANDLE hPipeWrite, const LAUNCH_PARAM& param) {
    zeno::log_trace("runner got program JSON: {}", progJson);

#if 0
    std::ofstream file("solver_serialize.json", std::ios::out | std::ios::binary);
    file.write(progJson.c_str(), progJson.size());
    file.close();
#endif

    //MessageBox(0, "runner", "runner", MB_OK);           //convient to attach process by debugger, at windows.
    zeno::scope_exit sp([=]() { std::cout.flush(); });
    //zeno::TimerAtexitHelper timerHelper;

    auto session = &zeno::getSession();
    session->globalState->sessionid = sessionid;
    session->globalState->clearState();
    session->globalComm->clearState();
    session->globalStatus->clearState();
    auto graph = session->createGraph();

    //$ZSG value
    zeno::setConfigVariable("ZSG", param.zsgPath.toStdString());
    //$FPS, getFrameTime value
    zeno::setConfigVariable("FPS", QString::number(param.projectFps).toStdString());

    zeno::getSession().globalComm->objTmpCachePath = param.objCacheDir.toStdString();

    float fps = param.projectFps;
    zeno::getSession().globalState->frame_time = (fps > 0) ? (1.f / fps) : 24;

    if (param.enableCache) {
        zeno::getSession().globalComm->frameCache(param.cacheDir.toStdString(), param.cacheNum);
    }
    else {
        zeno::getSession().globalComm->frameCache("", 0);
    }

    auto onfail = [&] {
        auto statJson = session->globalStatus->toJson();
        send_packet(hPipeWrite, "{\"action\":\"failed\"}");
        return 1;
    };

    zeno::GraphException::catched([&] {
        graph->loadGraph(progJson.c_str());
    }, *session->globalStatus);
    if (session->globalStatus->failed())
        return onfail();

    std::vector<char> buffer;

    session->globalComm->initFrameRange(graph->beginFrameNumber, graph->endFrameNumber);

    zeno::getSession().globalState->zeno_version = getZenoVersion();

    for (int frame = graph->beginFrameNumber; frame <= graph->endFrameNumber; frame++)
    {
        zeno::scope_exit sp([=]() { std::cout.flush(); });
        zeno::log_debug("begin frame {}", frame);

        session->globalState->frameid = frame;
        session->globalComm->newFrame();
        session->globalState->frameBegin();

        while (session->globalState->substepBegin())
        {
            zeno::GraphException::catched([&] {
                graph->applyNodesToExec();
            }, *session->globalStatus);
            session->globalState->substepEnd();
            if (session->globalStatus->failed())
                return onfail();
        }
        session->globalComm->finishFrame();

        zeno::log_debug("end frame {}", frame);

        if (param.enableCache) {
            //construct cache lock.
            std::string sLockFile = param.cacheDir.toStdString() + "/" + zeno::iotags::sZencache_lockfile_prefix + std::to_string(frame) + ".lock";
            QLockFile lckFile(QString::fromStdString(sLockFile));
            bool ret = lckFile.tryLock();
            //dump cache to disk.
            session->globalComm->dumpFrameCache(frame, param.applyLightAndCameraOnly, param.applyMaterialOnly);
            send_packet(hPipeWrite, "{\"action\":\"finishFrame\",\"key\":\"" + std::to_string(frame) + "\"}");
        } else {
            //后续可能支持直接回传内存的方式
            auto const& viewObjs = session->globalComm->getViewObjects();
            zeno::log_debug("runner got {} view objects", viewObjs.size());
            for (auto const& [key, obj] : viewObjs) {
                if (zeno::encodeObject(obj.get(), buffer)) {
                    send_packet(hPipeWrite, "{\"action\":\"viewObject\",\"key\":\"" + key + "\"}");
                }
                buffer.clear();
            }
        }
        if (session->globalStatus->failed())
            return onfail();
    }
    return 0;
}
