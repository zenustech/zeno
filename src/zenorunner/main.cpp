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

static void runner_main(std::string const &progJson) {
    zeno::log_debug("runner got program JSON: {}", progJson);

    auto session = &zeno::getSession();
    session->globalState->clearState();

    auto graph = session->createGraph();
    graph->loadGraph(progJson.c_str());

    std::vector<char> buffer;

    auto nframes = graph->adhocNumFrames;
    for (int frame = 0; frame < nframes; frame++) {
        zeno::log_debug("begin of frame {}", frame);
        session->globalState->frameBegin();
        while (session->globalState->substepBegin())
        {
            graph->applyNodesToExec();
            session->globalState->substepEnd();
        }

        auto viewObjs = session->globalState->globalComm->getViewObjects();
        zeno::log_debug("runner got {} view objects", viewObjs.size());
        session->globalState->frameEnd();
        zeno::log_debug("end of frame {}", frame);

        for (auto const &obj: viewObjs) {
            std::string info = "{\"action\":\"viewObject\"}";

            buffer.resize(4 + sizeof(Header) + info.size());
            buffer[0] = '\a';
            buffer[1] = '\b';
            buffer[2] = '\r';
            buffer[3] = '\t';
            std::memcpy(buffer.data() + 4 + sizeof(Header), info.data(), info.size());

            zeno::encodeObject(obj.get(), buffer);

            Header header;
            header.total_size = buffer.size();
            header.info_size = info.size();
            header.makeValid();
            std::memcpy(buffer.data() + 4, &header, sizeof(Header));

            zeno::log_debug("runner tx buffer size {}", buffer.size());
            for (char c: buffer) {
                fputc(c, old_stdout);
            }
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
