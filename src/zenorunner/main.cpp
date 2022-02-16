#include <cstdio>
#include <iostream>
#include <zeno/utils/log.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/zeno.h>
#include <string>

static FILE *g_pipe;

static void runner_main(std::string const &progJson) {
    zeno::log_debug("runner got program JSON: {}", progJson);

    auto session = &zeno::getSession();
    session->globalState->clearState();

    auto graph = session->createGraph();
    graph->loadGraph(progJson.c_str());

    auto nframes = graph->adhocNumFrames;
    for (int i = 0; i < nframes; i++) {
        session->globalState->frameBegin();
        while (session->globalState->substepBegin())
        {
            graph->applyNodesToExec();
            session->globalState->substepEnd();
        }
        session->globalState->frameEnd();
        // TODO: serialize globalComm and dump to g_pipe
    }
}

int main() {
    fprintf(stderr, "Zeno runner started...\n");
    fprintf(stdout, "(stdout ping test)\n");
    fflush(stderr);
    fflush(stdout);

    g_pipe = stdout;
    stdout = stderr;
    std::cout.rdbuf(std::cerr.rdbuf());

    std::string progJson;
    std::istreambuf_iterator<char> iit(std::cin.rdbuf()), eiit;
    std::back_insert_iterator<std::string> sit(progJson);
    std::copy(iit, eiit, sit);

    runner_main(progJson);
    return 0;
}
