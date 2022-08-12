#include <QApplication>
#include <QCommandLineParser> 
#include "zenoplayer.h"

#include "zenoapplication.h"
#include "style/zenostyle.h"

int calcFrameCountByAudio(std::string path, int fps);

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
            {"sample", "sample", "sample count"},
            {"pixel", "pixel", "set record image pixel"},
            {"path", "path", "record dir"},
            {"audio", "audio", "audio path"},
        });
        cmdParser.process(a);
        if (cmdParser.isSet("zsg"))
            param.sZsgPath = cmdParser.value("zsg"); 
        if (cmdParser.isSet("record"))
            param.bRecord = cmdParser.value("record").toLower() == "true" ? true : false;
        if (cmdParser.isSet("frame"))
            param.iFrame = cmdParser.value("frame").toInt();
        if (cmdParser.isSet("sample"))
            param.iSample = cmdParser.value("sample").toInt();
        else
            param.iSample = 256;
        if (cmdParser.isSet("pixel"))
            param.sPixel = cmdParser.value("pixel");
        if (cmdParser.isSet("path"))
            param.sPath = cmdParser.value("path");
        if (cmdParser.isSet("audio")) {
            param.audioPath = cmdParser.value("audio");
            if(!cmdParser.isSet("frame")) {
                int count = calcFrameCountByAudio(param.audioPath.toStdString(), 24);
                param.iFrame = count;
            }
        }
    }
    qDebug() << param.sPath << param.bRecord << param.iFrame << param.iSample << param.sPixel << param.sPath;
	ZenoPlayer w(param);
	w.show();
	return a.exec();
}
