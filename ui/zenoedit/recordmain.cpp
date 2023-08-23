#include "AudioFile.h"
#include "zeno/extra/assetDir.h"
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3.h"
#include <QApplication>
#include "zenomainwindow.h"
#include "zeno/core/Session.h"
#include "zeno/types/UserData.h"
#include "util/apphelper.h"
#include "launch/ztcpserver.h"
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include "launch/corelaunch.h"
#include <zeno/utils/log.h>
#include "common.h"
#include <rapidjson/document.h>

//#define DEBUG_DIRECTLY


static int calcFrameCountByAudio(std::string path, int fps) {
    //auto *pFlie;
    auto *pFile = strrchr(path.c_str(), '.');
    if (pFile != NULL) {
        if (strcmp(pFile, ".wav") == 0) {
            AudioFile<float> wav;
            wav.load(path);
            uint64_t ret = wav.getNumSamplesPerChannel();
            ret = fps * ret / wav.getSampleRate();
            return ret + 1;
        }

        else if (strcmp(pFile, ".mp3") == 0) {
            std::ifstream file(path, std::ios::binary);
            auto data = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            static mp3dec_t mp3d;
            mp3dec_init(&mp3d);
            mp3dec_frame_info_t info;

            float pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
            int mp3len = 0;
            int sample_len = 0;

            while (true) {
                int samples = mp3dec_decode_frame(&mp3d, data.data() + mp3len, data.size() - mp3len, pcm, &info);
                if (samples == 0) {
                    break;
                }
                sample_len += samples;
                mp3len += info.frame_bytes;
            }
            uint64_t ret = sample_len / info.channels;
            ret = ret * fps / info.hz;
            return ret + 1;
        }
    }
    return 0;
}

int record_main(const QCoreApplication& app);
int record_main(const QCoreApplication& app)
{
    //MessageBox(0, "recordcmd", "recordcmd", MB_OK);

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
        {"cacheautorm", "cacheautoremove", "remove cache after render"},
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
    LAUNCH_PARAM launchparam;
    QFileInfo fp(param.sZsgPath);
    launchparam.zsgPath = fp.absolutePath();
    if (cmdParser.isSet("cachePath")) {
        QString text = cmdParser.value("cachePath");
        text.replace('\\', '/');
        launchparam.cacheDir = text;
        launchparam.enableCache = true;
        launchparam.tempDir = false;
        if (!QDir(text).exists()) {
            QDir().mkdir(text);
        }
        if (cmdParser.isSet("cacheautorm"))
        {
            launchparam.autoRmCurcache = cmdParser.value("cacheautorm").toInt();
    }
        else {
            launchparam.autoRmCurcache = true;
        }
    if (cmdParser.isSet("cacheNum")) {
            launchparam.cacheNum = cmdParser.value("cacheNum").toInt();
    }
        else {
            launchparam.cacheNum = 1;
        }
    }
    else {
        zeno::log_info("cachePath missed, process exited with {}", -1);
        return -1;
        //launchparam.enableCache = true;
        //launchparam.tempDir = true;
    }
    if (cmdParser.isSet("exitWhenRecordFinish"))
        param.exitWhenRecordFinish = cmdParser.value("exitWhenRecordFinish").toLower() == "true";
    if (cmdParser.isSet("audio")) {
        param.audioPath = cmdParser.value("audio");
        if (!cmdParser.isSet("frame")) {
            int count = calcFrameCountByAudio(param.audioPath.toStdString(), 24);
            param.iFrame = count;
        }
    }
    param.iBitrate = cmdParser.isSet("bitrate") ? cmdParser.value("bitrate").toInt() : 20000;
    param.iFps = cmdParser.isSet("fps") ? cmdParser.value("fps").toInt() : 24;
	param.bOptix = cmdParser.isSet("optix") ? cmdParser.value("optix").toInt() : 0;
	param.isExportVideo = cmdParser.isSet("video") ? cmdParser.value("video").toInt() : 0;
	param.needDenoise = cmdParser.isSet("needDenoise") ? cmdParser.value("needDenoise").toInt() : 0;
	int enableAOV = cmdParser.isSet("aov") ? cmdParser.value("aov").toInt() : 0;
    auto &ud = zeno::getSession().userData();
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


    if (!param.bOptix) {
        //gl normal recording may not be work in cmd mode.
        ZenoMainWindow tempWindow(nullptr, 0, !param.bOptix ? PANEL_GL_VIEW : PANEL_EMPTY);
        tempWindow.showMaximized();
        tempWindow.solidRunRender(param,launchparam);

        //idle
        return app.exec();
    }
    else
    {
        QDir dir(param.sPath);
        if (!dir.exists())
        {
            zeno::log_info("output path does not exist, process exit with -1.");
            return -1;
        }
        else {
            dir.mkdir("P");
        }

        VideoRecInfo recInfo = AppHelper::getRecordInfo(param);

        //start a calc proc
        launchparam.beginFrame = recInfo.frameRange.first;
        launchparam.endFrame = recInfo.frameRange.second;

        bool ret = AppHelper::openZsgAndRun(param, launchparam);
        ZASSERT_EXIT(ret, -1); //will launch tcp server to start a calc proc.

        //get the final zencache path, like `2023-07-06 18-29-14`
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        QString zenCacheDir = mgr->cachePath();
        ZASSERT_EXIT(!zenCacheDir.isEmpty(), -1);
        QStringList args = QCoreApplication::arguments();

        int idxCachePath = args.indexOf("--cachePath");
        ZASSERT_EXIT(idxCachePath != -1 && idxCachePath + 1 < args.length(), -1);
        args[idxCachePath + 1] = zenCacheDir;

        auto pGraphs = zenoApp->graphsManagment();
        ZASSERT_EXIT(pGraphs, -1);

        ZASSERT_EXIT(args[1] == "--record", -1);
        args[1] = "--optixcmd";
        args[2] = QString::number(0);      //no need tcp
        args.append("--cacheautorm");
        args.append(QString::number(launchparam.autoRmCurcache));
        args.append("--optixShowBackground");
        args.append(QString::number(pGraphs->userdataInfo().optix_show_background));
        args.removeAt(0);

        //start optix proc to render
        QProcess* optixProc = new QProcess;

        optixProc->setInputChannelMode(QProcess::InputChannelMode::ManagedInputChannel);
        optixProc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
        bool enableOptixLog = true;
        if (enableOptixLog)
            optixProc->setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedErrorChannel);
        else
            optixProc->setProcessChannelMode(QProcess::ProcessChannelMode::SeparateChannels);
        optixProc->start(QCoreApplication::applicationFilePath(), args);

        if (!optixProc->waitForStarted(-1)) {
            zeno::log_warn("optix process failed to get started, giving up");
            return -1;
        }

        optixProc->closeWriteChannel();

        QObject::connect(optixProc, &QProcess::readyRead, [=]() {

            while (optixProc->canReadLine()) {
                QByteArray content = optixProc->readLine();

                if (content.startsWith("[optixcmd]:")) {
                    static const QString sFlag = "[optixcmd]:";
                    content = content.mid(sFlag.length());
                    rapidjson::Document doc;
                    doc.Parse(content);
                    ZASSERT_EXIT(doc.IsObject());
                    if (doc.HasMember("result")) {
                        ZASSERT_EXIT(doc["result"].IsInt());
                        int ret = doc["result"].GetInt();
                        std::cout << "\n[record] result is " << ret << "\n" << std::flush;
                        QCoreApplication::exit(ret);
                    }
                    else if (doc.HasMember("frame")) {
                        ZASSERT_EXIT(doc["frame"].IsInt());
                        int frame = doc["frame"].GetInt();
                        std::cout << "\n[record] frame " << frame << " recording is finished.\n" << std::flush;
                    }
                }
            }
        });

        QObject::connect(optixProc, &QProcess::errorOccurred, [=](QProcess::ProcessError error) {
            if (QProcess::Crashed == error) {
                std::cout << "\n[record] render process has crashed\n" << std::flush;
                QCoreApplication::exit(-2);
            }
        });

        //QObject::connect(optixProc, &QProcess::finished, [=](int exitCode, QProcess::ExitStatus exitStatus) {

        //});

        //idle
        return app.exec();
    }

}
