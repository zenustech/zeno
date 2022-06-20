#ifdef ZENO_MULTIPROCESS
#include <cstdio>
#include <iostream>
#include <zeno/utils/log.h>
#include <zeno/utils/Timer.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/zeno.h>
#include <string>
#ifdef ZENO_IPC_USE_TCP
#include <QTcpServer>
#include <QtWidgets>
#include <QTcpSocket>
#endif
#include <zeno/utils/scope_exit.h>
#include "corelaunch.h"

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

static void runner_start(std::string const &progJson, int sessionid) {
    zeno::log_debug("runner got program JSON: {}", progJson);
    //MessageBox(0, "runner", "runner", MB_OK);           //convient to attach process by debugger, at windows.
    zeno::scope_exit sp([=]() { std::cout.flush(); });
    zeno::TimerAtexitHelper timerHelper;

    auto session = &zeno::getSession();
    session->globalState->sessionid = sessionid;
    session->globalState->clearState();
    session->globalComm->clearState();
    session->globalStatus->clearState();
    auto graph = session->createGraph();

    auto onfail = [&] {
        auto statJson = session->globalStatus->toJson();
        send_packet("{\"action\":\"reportStatus\"}", statJson.data(), statJson.size());
    };

    graph->loadGraph(progJson.c_str());
    if (session->globalStatus->failed())
        return onfail();

    std::vector<char> buffer;

    session->globalComm->frameRange(graph->beginFrameNumber, graph->endFrameNumber);
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
            graph->applyNodesToExec();
            session->globalState->substepEnd();
            if (session->globalStatus->failed())
                return onfail();
        }
        session->globalComm->finishFrame();

        auto const &viewObjs = session->globalComm->getViewObjects();
        zeno::log_debug("runner got {} view objects", viewObjs.size());
        zeno::log_debug("end frame {}", frame);

        send_packet("{\"action\":\"newFrame\"}", "", 0);

        for (auto const &[key, obj]: viewObjs) {
            if (zeno::encodeObject(obj.get(), buffer))
                send_packet("{\"action\":\"viewObject\",\"key\":\"" + key + "\"}",
                        buffer.data(), buffer.size());
            buffer.clear();
        }

        send_packet("{\"action\":\"finishFrame\"}", "", 0);

        if (session->globalStatus->failed())
            return onfail();
    }
}

}

int runner_main(int sessionid);
int runner_main(int sessionid) {
    printf("(stdout ping test)\n");

#ifdef ZENO_IPC_USE_TCP
    clientSocket = std::make_unique<QTcpSocket>();
    clientSocket->connectToHost(QHostAddress::LocalHost, TCP_PORT);
    if (!clientSocket->waitForConnected(10000)) {
        zeno::log_error("tcp client connection fail.");
        return 0;
    } else {
        zeno::log_info("connect succeed!");
    }
#else
    ourfp = stdout;
#endif

    zeno::set_log_stream(std::cout);
    zeno::log_debug("runner started on sessionid={}", sessionid);

#if 0
#ifdef __linux__
    stdout = stderr;
#endif
    std::cout.rdbuf(std::cerr.rdbuf());
#endif

    std::string progJson;
    std::istreambuf_iterator<char> iit(std::cin.rdbuf()), eiit;
    std::back_insert_iterator<std::string> sit(progJson);
    std::copy(iit, eiit, sit);

    runner_start(progJson, sessionid);
    return 0;
}
#endif
