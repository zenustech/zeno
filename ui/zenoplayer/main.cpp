#include <QApplication>
#include <QCommandLineParser> 
#include "zenoplayer.h"

#include "zenoapplication.h"
#include "style/zenostyle.h"
#include <zeno/utils/logger.h>

namespace zaudio {
int calcFrameCountByAudio(std::string path, int fps);
}

int main(int argc, char *argv[]) 
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    a.setWindowIcon(QIcon(":/icons/zenus.png"));

    //QMessageBox::information(NULL, "debug", "debug");
	
    ZENO_PLAYER_INIT_PARAM param;
    param.init();
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
        });
        cmdParser.process(a);
        if (cmdParser.isSet("zsg"))
            param.sZsgPath = cmdParser.value("zsg"); 
        if (cmdParser.isSet("record"))
            param.bRecord = cmdParser.value("record").toLower() == "true" ? true : false;
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
        if (cmdParser.isSet("audio")) {
            param.audioPath = cmdParser.value("audio");
            if(!cmdParser.isSet("frame")) {
                int count = zaudio::calcFrameCountByAudio(param.audioPath.toStdString(), 24);
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
