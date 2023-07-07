#include "AudioFile.h"
#include "zeno/extra/assetDir.h"
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include <QApplication>
#include <QTcpServer>
#include <QtWidgets>
#include <QTcpSocket>
#include <zeno/utils/log.h>
#include "zenomainwindow.h"
#include "settings/zsettings.h"
#include "zeno/core/Session.h"
#include "zeno/types/UserData.h"
#include "viewport/optixviewport.h"
#include "util/apphelper.h"
#include "common.h"

//#define DEBUG_DIRECTLY

int optixcmd(const QCoreApplication& app, int port)
{
    ZENO_RECORD_RUN_INITPARAM param;
#ifndef DEBUG_DIRECTLY
    QCommandLineParser cmdParser;
    cmdParser.addHelpOption();
    cmdParser.addOptions({
        {"zsg", "zsg", "zsg file path"},
        {"record", "record", "Record frame"},
        {"frame", "frame", "frame count"},
        {"sframe", "sframe", "start frame"},
        {"sample", "sample", "sample count"},
        {"pixel", "pixel", "set record image pixel"},
        {"path", "path", "record dir"},
        {"audio", "audio", "audio path"},
        {"bitrate", "bitrate", "bitrate"},
        {"fps", "fps", "fps"},
        {"configFilePath", "configFilePath", "configFilePath"},
        {"cachePath", "cachePath", "cachePath"},
        {"cacheNum", "cacheNum", "cacheNum"},
        {"exitWhenRecordFinish", "exitWhenRecordFinish", "exitWhenRecordFinish"},
        {"optix", "optix", "optix mode"},
        {"video", "video", "export video"},
        {"aov", "aov", "aov"},
        {"needDenoise", "needDenoise", "needDenoise"},
        {"videoname", "videoname", "export video's name"},
        {"subzsg", "subgraphzsg", "subgraph zsg file path"},
        });
    cmdParser.process(app);

    if (cmdParser.isSet("zsg"))
        param.sZsgPath = cmdParser.value("zsg");
    if (cmdParser.isSet("record"))
        param.bRecord = cmdParser.value("record").toLower() == "true";
    if (cmdParser.isSet("frame"))
        param.iFrame = cmdParser.value("frame").toInt();
    if (cmdParser.isSet("sframe"))
        param.iSFrame = cmdParser.value("sframe").toInt();
    if (cmdParser.isSet("sample"))
        param.iSample = cmdParser.value("sample").toInt();
    if (cmdParser.isSet("pixel"))
        param.sPixel = cmdParser.value("pixel");
    if (cmdParser.isSet("path"))
        param.sPath = cmdParser.value("path");
    if (cmdParser.isSet("configFilePath")) {
        param.configFilePath = cmdParser.value("configFilePath");
        zeno::setConfigVariable("configFilePath", param.configFilePath.toStdString());
    }
    if (cmdParser.isSet("cachePath")) {
        QString text = cmdParser.value("cachePath");
        text.replace('\\', '/');
        if (!QDir(text).exists()) {
            QDir().mkdir(text);
        }
    }
    if (cmdParser.isSet("cacheNum")) {
    }
    if (cmdParser.isSet("exitWhenRecordFinish"))
        param.exitWhenRecordFinish = cmdParser.value("exitWhenRecordFinish").toLower() == "true";
    if (cmdParser.isSet("audio")) {
        param.audioPath = cmdParser.value("audio");
        //todo: resolve include compile problem(zeno\tpls\include\minimp3.h).
        //if (!cmdParser.isSet("frame")) {
        //    int count = calcFrameCountByAudio(param.audioPath.toStdString(), 24);
        //    param.iFrame = count;
        //}
    }
    param.iBitrate = cmdParser.isSet("bitrate") ? cmdParser.value("bitrate").toInt() : 20000;
    param.iFps = cmdParser.isSet("fps") ? cmdParser.value("fps").toInt() : 24;
    param.bOptix = cmdParser.isSet("optix") ? cmdParser.value("optix").toInt() : 0;
    param.isExportVideo = cmdParser.isSet("video") ? cmdParser.value("video").toInt() : 0;
    param.needDenoise = cmdParser.isSet("needDenoise") ? cmdParser.value("needDenoise").toInt() : 0;
    int enableAOV = cmdParser.isSet("aov") ? cmdParser.value("aov").toInt() : 0;
    auto& ud = zeno::getSession().userData();
    ud.set2("output_aov", enableAOV != 0);
    param.videoName = cmdParser.isSet("videoname") ? cmdParser.value("videoname") : "output.mp4";
    param.subZsg = cmdParser.isSet("subzsg") ? cmdParser.value("subzsg") : "";
#else
    param.sZsgPath = "C:\\zeno\\framenum.zsg";
    param.sPath = "C:\\recordpath";
    param.iFps = 24;
    param.iBitrate = 200000;
    param.iSFrame = 0;
    param.iFrame = 10;
    param.iSample = 1;
    param.bOptix = true;
    param.sPixel = "1200x800";
#endif

    VideoRecInfo recInfo = AppHelper::getRecordInfo(param);

    QTcpSocket optixClientSocket;
    optixClientSocket.connectToHost(QHostAddress::LocalHost, port);
    if (!optixClientSocket.waitForConnected(10000))
    {
        zeno::log_error("tcp optix client connection fail");
        return -1;
    }
    else
    {
        zeno::log_info("tcp optix client connection succeed");
    }

    OptixWorker worker;

    QObject::connect(&optixClientSocket, &QTcpSocket::readyRead, [&]() {
        QByteArray arr = optixClientSocket.readAll();
        qint64 redSize = arr.size();
        if (redSize > 0) {

            zeno::log_info("finish frame");
        }
    });

    return app.exec();
}
