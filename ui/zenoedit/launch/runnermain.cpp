#ifdef ZENO_MULTIPROCESS
#include <cstdio>
#include <iostream>
#include <zeno/utils/log.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/zeno.h>
#include <string>

namespace {

static FILE *old_stdout;
static char stdout_buf[1<<20]; // 1MB

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

static void send_packet(std::string const &info, const char *buf, size_t len) {
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

    zeno::log_debug("runner tx head-buffer size {}", headbuffer.size());
    for (char c: headbuffer) {
        fputc(c, old_stdout);
    }

    zeno::log_debug("runner tx data-buffer size {}", len);
    for (size_t i = 0; i < len; i++) {
        fputc(buf[i], old_stdout);
    }
}

static void runner_start(std::string const &progJson) {
    zeno::log_debug("runner got program JSON: {}", progJson);

    setvbuf(old_stdout, stdout_buf, _IOFBF, sizeof(stdout_buf));

    auto session = &zeno::getSession();
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

    for (int frame = graph->beginFrameNumber; frame < graph->endFrameNumber; frame++) {
        zeno::log_info("begin frame {}", frame);

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

        auto viewObjs = session->globalComm->getViewObjects();
        zeno::log_debug("runner got {} view objects", viewObjs.size());
        session->globalState->frameEnd();
        zeno::log_debug("end frame {}", frame);

        send_packet("{\"action\":\"newFrame\"}", "", 0);

        for (auto const &obj: viewObjs) {
            if (zeno::encodeObject(obj.get(), buffer))
                send_packet("{\"action\":\"viewObject\"}", buffer.data(), buffer.size());
            buffer.clear();
        }

        send_packet("{\"action\":\"finishFrame\"}", "", 0);

        if (session->globalStatus->failed())
            return onfail();
    }
}

}

int runner_main();
int runner_main() {
    fprintf(stderr, "Zeno runner started...\n");
    fprintf(stdout, "(stdout ping test)\n");

    old_stdout = stdout;
#ifdef __linux__
    stdout = stderr;
#else
    //(void)freopen("/dev/stderr", "w", stdout);//todo
#endif
    std::cout.rdbuf(std::cerr.rdbuf());

    std::string progJson;
    std::istreambuf_iterator<char> iit(std::cin.rdbuf()), eiit;
    std::back_insert_iterator<std::string> sit(progJson);
    std::copy(iit, eiit, sit);

    runner_start(progJson);
    return 0;
}
#endif
