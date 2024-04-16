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
#include <zeno/extra/assetDir.h>
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
#include "viewdecode.h"
#include "settings/zsettings.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include "startup/zstartup.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>
#include <zenomodel/include/jsonhelper.h>

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

static int runner_start(std::string const &progJson, int sessionid, const LAUNCH_PARAM& param) {
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
    session->globalComm->setSendPacketFunction(send_packet);

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
        send_packet("{\"action\":\"reportStatus\"}", statJson.data(), statJson.size());
        return 1;
    };

    zeno::GraphException::catched([&] {
        graph->loadGraph(progJson.c_str());
    }, *session->globalStatus);
    if (session->globalStatus->failed())
        return onfail();

    std::vector<char> buffer;

    session->globalComm->initFrameRange(graph->beginFrameNumber, graph->endFrameNumber);
    send_packet("{\"action\":\"frameRange\",\"key\":\""
                + std::to_string(graph->beginFrameNumber)
                + ":" + std::to_string(graph->endFrameNumber)
                + "\"}", "", 0);

    zeno::getSession().globalState->zeno_version = getZenoVersion();
    bool bHasDumpStatic = false;

    if (!param.generator.isEmpty())
    {
        //only execute the node which id is `param.generator`.
        std::set<std::string> nodes;
        std::string ident = param.generator.toStdString();
        nodes.insert(ident);
        graph->applyNodes(nodes);

        //yield result from the GenerateCommands node.
        const auto& node = graph->getNode(ident);
        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        {
            JsonObjBatch objBatch(writer);
            for (auto const& [ds, bound] : node->inputBounds) {
                const auto& sourceInput = graph->getNodeInput(ident, ds);
                if (ident.find("PythonNode") > 0 && ds == "args")
                {
                    auto dict = std::dynamic_pointer_cast<zeno::DictObject>(sourceInput);
                    writer.Key("args");
                    {
                        JsonObjBatch argBatch(writer);
                        int idx = 0;
                        for (const auto& [key, obj] : dict->lut)
                        {
                            std::shared_ptr<zeno::NumericObject> num = std::dynamic_pointer_cast<zeno::NumericObject>(obj);
                            if (num) {
                                writer.Key(key.c_str());
                                if (std::get_if<float>(&num->value))
                                {
                                    writer.Double(num->get<float>());
                                }
                                else if (std::get_if<int>(&num->value))
                                {
                                    writer.Int(num->get<int>());
                                }
                                else
                                {
                                    writer.Null();
                                    zeno::log_error("parse obj error");
                                }
                                idx++;
                            }
                            else
                            {
                                std::shared_ptr<zeno::StringObject> pStr = std::dynamic_pointer_cast<zeno::StringObject>(obj);
                                if (pStr) {
                                    writer.Key(key.c_str());
                                    writer.String(pStr->get().c_str());
                                    idx++;
                                }
                            }
                        }
                    }
                }
                else
                {
                    const auto& strObj = std::dynamic_pointer_cast<zeno::StringObject>(sourceInput);
                    if (strObj && !strObj->get().empty())
                    {
                        writer.Key(ds.c_str());
                        std::string commands = strObj->get();
                        writer.String(commands.c_str());
                    }
                }
            }
        }
        std::string strJson = s.GetString();
        if (strJson != "")
        {
            send_packet("{\"action\":\"generate\",\"key\":\"" + ident + "\"" + "}", strJson.c_str(), strJson.length() + 1);
            return 0;
        }
        //and then send packet back to ui process.
        return onfail();
    }

    for (int frame = graph->beginFrameNumber; frame <= graph->endFrameNumber; frame++)
    {
        zeno::scope_exit sp([=]() { std::cout.flush(); });
        zeno::log_debug("begin frame {}", frame);

        session->globalState->frameid = frame;
        session->globalComm->newFrame();
        session->globalState->frameBegin();

        ////construct cache lock.
        std::string sLockFile;
        if (!bHasDumpStatic && param.enableCache)
            sLockFile = param.cacheDir.toStdString() + "/" + zeno::iotags::sZencache_lockfile_prefix + "_static" + ".lock";
        if (param.enableCache)
            sLockFile = param.cacheDir.toStdString() + "/" + zeno::iotags::sZencache_lockfile_prefix + std::to_string(frame) + ".lock";
        QLockFile lckFile(QString::fromStdString(sLockFile));
        bool ret = lckFile.tryLock();

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

        send_packet("{\"action\":\"newFrame\",\"key\":\"" + std::to_string(frame) +"\"}", "", 0);

        if (!bHasDumpStatic && param.enableCache) {
            //dump cache to disk.
            session->globalComm->dumpFrameCache(-1);
            bHasDumpStatic = true;
        }

        if (param.enableCache) {
            //dump cache to disk.
            session->globalComm->dumpFrameCache(frame);
        } else {
            const auto& objs = session->globalComm->pairsShared();
            zeno::log_debug("runner got {} view objects", objs.size());
            for (auto const& p : objs) {
                if (zeno::encodeObject(p.second.get(), buffer))
                    send_packet("{\"action\":\"viewObject\",\"key\":\"" + p.first + "\"}",
                        buffer.data(), buffer.size());
                buffer.clear();
            }
        }
        lckFile.unlock();

        send_packet("{\"action\":\"finishFrame\",\"key\":\"" + std::to_string(frame) + "\"}", "", 0);

        if (session->globalStatus->failed())
            return onfail();
    }

    return 0;
}

}
int runner_main(const QCoreApplication& app);
int runner_main(const QCoreApplication& app) {
    //MessageBox(0, "runner", "runner", MB_OK);           //convient to attach process by debugger, at windows.

#ifdef __linux__
    stderr = freopen("/dev/stdout", "w", stderr);
#endif
    LAUNCH_PARAM param;
    int sessionid = 0;
    int port = -1;
    std::string objcachedir = "";
    QCommandLineParser cmdParser;
    cmdParser.addHelpOption();
    cmdParser.addOptions({
        {"runner", "runner", "runner"},
        {"sessionid", "sessionid", "sessionid"},
        {"port", "port", "tcp server port"},
        {"enablecache", "enablecache", "enable zencache"},
        {"cachenum", "cachenum", "max cached frames"},
        {"cachedir", "cachedir", "cache dir for this run"},
        {"cacheautorm", "cacheautoremove", "remove cache after render"},
        {"zsg", "zsg", "zsg"},
        {"projectFps", "current project fps", "fps"},
        {"objcachedir", "objcachedir", "obj temp cache dir"},
        {"generator", "generator", "the node ident which trigger generate command"},
        });
    cmdParser.process(app);
    if (cmdParser.isSet("sessionid"))
        sessionid = cmdParser.value("sessionid").toInt();
    if (cmdParser.isSet("port"))
        port = cmdParser.value("port").toInt();
    if (cmdParser.isSet("enablecache"))
        param.enableCache = cmdParser.value("enablecache").toInt();
    if (cmdParser.isSet("cachenum"))
        param.cacheNum = cmdParser.value("cachenum").toInt();
    if (cmdParser.isSet("cachedir"))
        param.cacheDir = cmdParser.value("cachedir");
    if (cmdParser.isSet("objcachedir"))
        param.objCacheDir = cmdParser.value("objcachedir");
    if (cmdParser.isSet("cacheautorm"))
        param.autoRmCurcache = cmdParser.value("cacheautorm").toInt();
    if (cmdParser.isSet("zsg"))
        param.zsgPath = cmdParser.value("zsg");
    if (cmdParser.isSet("projectFps"))
        param.projectFps = cmdParser.value("projectFps").toInt();
    if (cmdParser.isSet("generator"))
        param.generator = cmdParser.value("generator");

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

    return runner_start(progJson, sessionid, param);
}
#endif
