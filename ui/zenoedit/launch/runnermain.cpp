#ifdef ZENO_MULTIPROCESS
#include <cstdio>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <zeno/utils/log.h>
#include <zeno/utils/Timer.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/GraphException.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/zeno.h>
#include <string>
#ifdef ZENO_IPC_USE_TCP
#include <QTcpServer>
#include <QtWidgets>
#include <QTcpSocket>
#include <QSettings>
#endif
#include <zeno/utils/scope_exit.h>
#include "corelaunch.h"
#include "viewdecode.h"
#include "settings/zsettings.h"

namespace {

#ifdef ZENO_IPC_USE_TCP
static std::unique_ptr<QTcpSocket> clientSocket;
#else
static FILE *ourfp;
static char ourbuf[1 << 20]; // 1MB
#endif

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

static void send_packet(std::string_view info, const char *buf, size_t len) {
    Header header;
    header.total_size = info.size() + len;
    header.info_size = info.size();
    header.makeValid();

    std::vector<char> headbuffer(4 + sizeof(Header) + info.size());
    headbuffer[0] = '\a';
    headbuffer[1] = '\b';
    headbuffer[2] = '\r';
    headbuffer[3] = '\t';
    std::memcpy(headbuffer.data() + 4, &header, sizeof(Header));
    std::memcpy(headbuffer.data() + 4 + sizeof(Header), info.data(), info.size());

    zeno::log_debug("runner tx head-buffer {} data-buffer {}", headbuffer.size(), len);
#ifdef ZENO_IPC_USE_TCP
    for (char c: headbuffer) {
        clientSocket->write(&c, 1);
    }
    clientSocket->write(buf, len);
    while (clientSocket->bytesToWrite() > 0) {
        clientSocket->waitForBytesWritten();
    }
#else
    for (char c : headbuffer) {
        fputc(c, ourfp);
    }
    for (size_t i = 0; i < len; i++) {
        fputc(buf[i], ourfp);
    }
    fflush(ourfp);
#endif
}

static size_t getCacheDirsize(std::string cachedir) {
    size_t cacheDirSize = 0;
    for (auto& direntry : std::filesystem::recursive_directory_iterator(std::filesystem::u8path(cachedir)))
    {
        if (!std::filesystem::is_directory(direntry.path()))
            cacheDirSize += std::filesystem::file_size(direntry.path());
    }
    return cacheDirSize;
}

static int runner_start(std::string const &progJson, int sessionid, char* cachedir, bool cacheLightCameraOnly, bool cacheMaterialOnly) {
    zeno::log_trace("runner got program JSON: {}", progJson);
    //MessageBox(0, "runner", "runner", MB_OK);           //convient to attach process by debugger, at windows.
    zeno::scope_exit sp([=]() { std::cout.flush(); });
    //zeno::TimerAtexitHelper timerHelper;

    auto session = &zeno::getSession();
    session->globalState->sessionid = sessionid;
    session->globalState->clearState();
    session->globalComm->clearState();
    session->globalStatus->clearState();
    auto graph = session->createGraph();

    auto onfail = [&] {
        auto statJson = session->globalStatus->toJson();
        send_packet("{\"action\":\"reportStatus\"}", statJson.data(), statJson.size());
        return 1;
    };

    zeno::GraphException::catched([&] {
        graph->loadGraph(progJson.c_str());
    }, *session->globalStatus);
    if (session->globalStatus->failed())
        return onfail();

    bool bZenCache = initZenCache(cachedir);

    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    size_t cacheDirSize = 0;
    if (bZenCache && (settings.value("zencache-rmcurcache").isValid() ? settings.value("zencache-rmcurcache").toBool() : false))
        cacheDirSize = getCacheDirsize(std::filesystem::u8path(cachedir).parent_path().string());

    std::vector<char> buffer;

    session->globalComm->initFrameRange(graph->beginFrameNumber, graph->endFrameNumber);
    send_packet("{\"action\":\"frameRange\",\"key\":\""
                + std::to_string(graph->beginFrameNumber)
                + ":" + std::to_string(graph->endFrameNumber)
                + "\"}", "", 0);

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

        send_packet("{\"action\":\"newFrame\"}", "", 0);

        if (bZenCache) {
            if (settings.value("zencache-rmcurcache").isValid() ? settings.value("zencache-rmcurcache").toBool() : false)
            {
                while (cacheDirSize > 214748364800) //200GB
                {
                    zeno::log_info("cacheDirSize {} byte, wait for zencache remove", cacheDirSize);
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    cacheDirSize = getCacheDirsize(std::filesystem::u8path(cachedir).parent_path().string());
                }
            }
            session->globalComm->dumpFrameCache(frame, cacheDirSize, cacheLightCameraOnly, cacheMaterialOnly);
        } else {
            auto const& viewObjs = session->globalComm->getViewObjects();
            zeno::log_debug("runner got {} view objects", viewObjs.size());
            for (auto const& [key, obj] : viewObjs) {
                if (zeno::encodeObject(obj.get(), buffer))
                    send_packet("{\"action\":\"viewObject\",\"key\":\"" + key + "\"}",
                        buffer.data(), buffer.size());
                buffer.clear();
            }
        }

        send_packet("{\"action\":\"finishFrame\"}", "", 0);

        if (session->globalStatus->failed())
            return onfail();
    }
    return 0;
}

}

int runner_main(int sessionid, int port, char *cachedir, bool cacheLightCameraOnly, bool cacheMaterialOnly);
int runner_main(int sessionid, int port, char *cachedir, bool cacheLightCameraOnly, bool cacheMaterialOnly) {
#ifdef __linux__
    stderr = freopen("/dev/stdout", "w", stderr);
#endif
    std::cerr.rdbuf(std::cout.rdbuf());
    std::clog.rdbuf(std::cout.rdbuf());

    zeno::set_log_stream(std::clog);

#ifdef ZENO_IPC_USE_TCP
    zeno::log_debug("connecting to port {}", port);
    clientSocket = std::make_unique<QTcpSocket>();
    clientSocket->connectToHost(QHostAddress::LocalHost, port);
    if (!clientSocket->waitForConnected(10000)) {
        zeno::log_error("tcp client connection fail");
        return 0;
    } else {
        zeno::log_info("tcp connection succeed");
    }
#else
    zeno::log_debug("started IPC in pipe mode");
    ourfp = stdout;
#endif

    zeno::log_debug("runner started on sessionid={}", sessionid);

    std::string progJson;
    std::istreambuf_iterator<char> iit(std::cin.rdbuf()), eiit;
    std::back_insert_iterator<std::string> sit(progJson);
    std::copy(iit, eiit, sit);


#ifdef ZENO_IPC_USE_TCP
    // Notify this is runner process
    static int calledOnce = ([]{
      zeno::getSession().eventCallbacks->triggerEvent("preRunnerStart");
    }(), 0);
#endif

    return runner_start(progJson, sessionid, cachedir, cacheLightCameraOnly, cacheMaterialOnly);
}
#endif
