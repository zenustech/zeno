#include <QApplication>
#include <QCommandLineParser>
#include "zenoplayer.h"
#include "zenoapplication.h"
#include "style/zenostyle.h"
#include "zeno/extra/assetDir.h"
#include <zeno/utils/logger.h>
#include "AudioFile.h"
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3.h"

static int calcFrameCountByAudio(std::string path, int fps) {
    //auto *pFlie;
    auto *pFile = strrchr(path.c_str(),'.');
    if(pFile !=NULL) {
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
}

int main(int argc, char *argv[])
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    a.setWindowIcon(QIcon(":/icons/zenus.png"));

    //QMessageBox::information(NULL, "debug", "debug");

    ZENO_PLAYER_INIT_PARAM param;
    if (argc > 1)
    {
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
                                 {"exitWhenRecordFinish", "exitWhenRecordFinish", "exitWhenRecordFinish"},
                             });
        cmdParser.process(a);
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
        if (cmdParser.isSet("exitWhenRecordFinish"))
            param.exitWhenRecordFinish = cmdParser.value("exitWhenRecordFinish").toLower() == "true";
        if (cmdParser.isSet("audio")) {
            param.audioPath = cmdParser.value("audio");
            if(!cmdParser.isSet("frame")) {
                int count = calcFrameCountByAudio(param.audioPath.toStdString(), 24);
                param.iFrame = count;
            }
        }
        param.iBitrate = cmdParser.isSet("bitrate")? cmdParser.value("bitrate").toInt(): 20000;
        param.iFps = cmdParser.isSet("fps")? cmdParser.value("fps").toInt(): 24;
    }
    zeno::log_info("ZsgPath {} Record {} Frame {} SFrame {} Sample {} Pixel {} Path {}",
                   param.sZsgPath.toStdString(), param.bRecord, param.iFrame, param.iSFrame, param.iSample,
                   param.sPixel.toStdString(), param.sPath.toStdString());
    ZenoPlayer w(param);
    w.show();
    return a.exec();
}
