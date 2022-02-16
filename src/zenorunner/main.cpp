#include <cstdio>
#include <iostream>
#include <zeno/utils/log.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/zeno.h>
#include <string>

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

static void send_packet(std::string const &info, std::vector<char> const &buffer) {
    Header header;
    header.total_size = info.size() + buffer.size();
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

    zeno::log_debug("runner tx data-buffer size {}", buffer.size());
    for (char c: buffer) {
        fputc(c, old_stdout);
    }
}

static void runner_main(std::string const &progJson) {
    zeno::log_debug("runner got program JSON: {}", progJson);

    setvbuf(old_stdout, stdout_buf, _IOFBF, sizeof(stdout_buf));

    auto session = &zeno::getSession();
    session->globalState->clearState();

    auto graph = session->createGraph();
    graph->loadGraph(progJson.c_str());

    std::vector<char> buffer;

    auto nframes = graph->adhocNumFrames;
    for (int frame = 0; frame < nframes; frame++) {
        zeno::log_info("begin frame {}", frame);
        session->globalState->frameBegin();
        session->globalComm->newFrame();
        while (session->globalState->substepBegin())
        {
            graph->applyNodesToExec();
            session->globalState->substepEnd();
        }

        auto viewObjs = session->globalComm->getViewObjects();
        zeno::log_debug("runner got {} view objects", viewObjs.size());
        session->globalState->frameEnd();
        zeno::log_debug("end frame {}", frame);

        send_packet("{\"action\":\"newFrame\"}", {});

        for (auto const &obj: viewObjs) {
            zeno::encodeObject(obj.get(), buffer);
            send_packet("{\"action\":\"viewObject\"}", buffer);
            buffer.clear();
        }
    }
}

int main() {
    fprintf(stderr, "Zeno runner started...\n");
    fprintf(stdout, "(stdout ping test)\n");

    old_stdout = stdout;
    stdout = stderr;
    std::cout.rdbuf(std::cerr.rdbuf());

    std::string progJson;
    std::istreambuf_iterator<char> iit(std::cin.rdbuf()), eiit;
    std::back_insert_iterator<std::string> sit(progJson);
    std::copy(iit, eiit, sit);

    runner_main(progJson);
    return 0;
}
