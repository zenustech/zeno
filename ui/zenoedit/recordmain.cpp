#include "AudioFile.h"
#include "zeno/extra/assetDir.h"
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3.h"
#include <QApplication>
#include "zenomainwindow.h"
#include "zeno/core/Session.h"
#include "zeno/types/UserData.h"
#include "launch/corelaunch.h"
#include <zeno/utils/log.h>

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

    ZenoMainWindow tempWindow(nullptr, 0, param.bOptix ? PANEL_OPTIX_VIEW : PANEL_GL_VIEW);
    if (!param.bOptix)
    {
        tempWindow.showMaximized();
        tempWindow.solidRunRender(param);
    }
    else {
        tempWindow.optixRunRender(param, launchparam);
    }
    return app.exec();
}
