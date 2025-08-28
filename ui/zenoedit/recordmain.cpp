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


//--record true --zsg "C:\zeno-master\render_param.zsg" --cachePath "C:\tmp" --sframe 0 --frame 10 --sample 1 --optix 0 --path "C:\recordpath" --pixel 4500x3500 --aov 0 --needDenoise 0
//--record true --zsg "C:\zeno-master\render_param.zsg" --cachePath "C:\tmp" --optix 1

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
        {"bitrate", "bitrate", "bitrate", "2000"},
        {"fps", "fps", "fps", "24"},
        {"configFilePath", "configFilePath", "configFilePath"},
        {"cachePath", "cachePath", "cachePath"},
        {"cacheNum", "cacheNum", "cacheNum"},
        {"exitWhenRecordFinish", "exitWhenRecordFinish", "exitWhenRecordFinish"},
        {"optix", "optix", "optix mode", "0"},
        {"video", "video", "export video", "0"},
        {"aov", "aov", "aov", "0"},
        {"exr", "exr", "exr", "0"},
        {"needDenoise", "needDenoise", "needDenoise", "0"},
        {"videoname", "videoname", "export video's name", "output.mp4"},
        {"subzsg", "subgraphzsg", "subgraph zsg file path", ""},
        {"cacheautorm", "cacheautoremove", "remove cache after render"},
        {"clearHistoryCache", "clearHistoryCache", "remove all history cache before run"},
        {"paramsPath", "paramsPath", "paramsPath"},
        {"paramsBase64", "paramsBase64", "paramsBase64"},
        {"paramsJson", "paramsJson", "paramsJson"},
        });
    cmdParser.process(app);

    if (cmdParser.isSet("zsg"))
    {
        param.sZsgPath = cmdParser.value("zsg");
        //�Ƚ���zsg�������Ⱦ����
        ZsgReader::getInstance().readRenderSettings(param.sZsgPath, param);
    }
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
    launchparam.fromCmd = true;
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
    if (cmdParser.isSet("paramsPath"))
    {
        launchparam.paramPath = cmdParser.value("paramsPath");
    }
    if (cmdParser.isSet("paramsBase64"))
    {
        launchparam.paramBase64 = cmdParser.value("paramsBase64");
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
    if (cmdParser.isSet("clearHistoryCache")) {
        launchparam.cmdRmHistoryCacheBeforeRun = cmdParser.value("clearHistoryCache").toInt();
    }

    //parse render params:
    if (cmdParser.isSet("bitrate")) {
        param.iBitrate = cmdParser.value("bitrate").toInt();
    }
    if (cmdParser.isSet("fps")) {
        param.iFps = cmdParser.value("fps").toInt();
    }
    if (cmdParser.isSet("optix")) {
        param.bOptix = cmdParser.value("optix").toInt();
    }
    if (cmdParser.isSet("video")) {
        param.isExportVideo = cmdParser.value("video").toInt();
    }
    if (cmdParser.isSet("needDenoise")) {
        param.needDenoise = cmdParser.value("needDenoise").toInt();
    }
    if (cmdParser.isSet("aov")) {
        param.bAov = cmdParser.value("aov").toInt();
        auto& ud = zeno::getSession().userData();
        ud.set2("output_aov", param.bAov != 0);
    }
    if (cmdParser.isSet("exr")) {
        param.export_exr = cmdParser.value("exr").toInt() != 0;
        auto& ud = zeno::getSession().userData();
        ud.set2("output_exr", param.export_exr);
    }
    if (cmdParser.isSet("videoname")) {
        param.videoName = cmdParser.value("videoname");
    }
    if (cmdParser.isSet("subzsg")) {
        param.subZsg = cmdParser.value("subzsg");
    }

    if (cmdParser.isSet("paramsJson"))
    {
        param.paramsJson = cmdParser.value("paramsJson");        
    }

    if (!param.bOptix) {
        ZenoMainWindow tempWindow(nullptr, 0, !param.bOptix ? PANEL_GL_VIEW : PANEL_EMPTY);
        tempWindow.hide();
        tempWindow.solidRunRender(param, launchparam);
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

        //start optix proc to render
        QProcess* optixProc = new QProcess;

        QObject::connect(zenoApp->getServer(), &ZTcpServer::runnerError, [=]() {
            std::cout << "\n[record] calculation process has error and exit.\n" << std::flush;
            optixProc->kill();
            QCoreApplication::exit(-2);
        });

        QObject::connect(zenoApp->getServer(), &ZTcpServer::runFinished, [=]() {
            if (!param.bRecord) {
                std::cout << "\n[record] record set to false. calculation process finished and exit.\n" << std::flush;
                QCoreApplication::exit(0);
            }
            if (param.sPath.isEmpty()) {
                std::cout << "\n[record] record failed because the path to store image has not been set." << std::flush;
                QCoreApplication::exit(-1);
            }
        });

        bool ret = AppHelper::openZsgAndRun(param, launchparam);
        ZERROR_EXIT(ret, -1); //will launch tcp server to start a calc proc.

        if (param.bRecord && !param.sPath.isEmpty()) {
            //get the final zencache path, like `2023-07-06 18-29-14`
            std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
            QString zenCacheDir = mgr->cachePath();
            ZERROR_EXIT(!zenCacheDir.isEmpty(), -1);

            QStringList cmdArgs = QCoreApplication::arguments();
            QStringList args;

            int idxCachePath = cmdArgs.indexOf("--cachePath");
            if (idxCachePath == -1)
            {
                zeno::log_error("no cache path offered, please specifiy with --cachePath");
                return -1;
            }

            auto pGraphs = zenoApp->graphsManagment();
            ZERROR_EXIT(pGraphs, -1);

            args.append("--optixcmd");
            args.append("0");

            args.append("--zsg");
            args.append(param.sZsgPath);

            args.append("--cachePath");
            args.append(zenCacheDir);

            args.append("--cacheautorm");
            args.append(QString::number(launchparam.autoRmCurcache));

            args.append("--optixShowBackground");
            args.append(QString::number(pGraphs->userdataInfo().optix_show_background));

            args.append("--frame");
            args.append(QString::number(param.iFrame));

            args.append("--sframe");
            args.append(QString::number(param.iSFrame));

            args.append("--sample");
            args.append(QString::number(param.iSample));

            args.append("--pixel");
            args.append(param.sPixel);

            args.append("--path");
            args.append(param.sPath);

            args.append("--audio");
            args.append(param.audioPath);

            args.append("--bitrate");
            args.append(QString::number(param.iBitrate));

            args.append("--fps");
            args.append(QString::number(param.iFps));

            args.append("--configFilePath");
            args.append(param.configFilePath);

            args.append("--cacheNum");
            args.append(QString::number(launchparam.cacheNum));

            args.append("--exitWhenRecordFinish");
            args.append(QString::number(param.exitWhenRecordFinish));

            args.append("--optix");
            args.append(QString::number(param.bOptix));

            args.append("--video");
            args.append(QString::number(param.isExportVideo));

            args.append("--aov");
            args.append(QString::number(param.bAov));

            args.append("--exr");
            args.append(QString::number(param.export_exr));

            args.append("--needDenoise");
            args.append(QString::number(param.needDenoise));

            args.append("--videoname");
            args.append(param.videoName);

            args.append("--subzsg");
            args.append(param.subZsg);

            args.append("--paramsPath");
            args.append(launchparam.paramPath);

            args.append("--paramsJson");
            args.append(param.paramsJson);

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
                            std::string err_info;
                            if (ret != 0) {
                                switch (ret)
                                {
                                case REC_NO_RECORD_OPTION:
                                    err_info = "[error]no record video, please set video=1 on command line";
                                    break;
                                case REC_OPTIX_INTERNAL_FATAL:
                                    err_info = "[error]internal error on ";
                                    break;
                                case REC_FFMPEG_FATAL:
                                    err_info = "[error]run error on ffmpeg command";
                                    break;
                                case REC_NOFFMPEG:
                                    err_info = "[error]no ffmpeg.exe deployed on dir same as zeno.";
                                    break;
                                }
                            }
                            else {
                                err_info = "the recording returns no error;";
                            }
                            std::cout << "\n" << err_info << "\n" << std::flush;
                            QCoreApplication::exit(ret);
                        }
                        else if (doc.HasMember("frame")) {
                            ZASSERT_EXIT(doc["frame"].IsInt());
                            int frame = doc["frame"].GetInt();
                            std::cout << "\n[record] frame " << frame << " recording is finished.\nProgress: "<< (int)(100 * (float)(frame+1-param.iSFrame)/(float)param.iFrame) <<"%\n"<< std::flush;
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
        }

        //QObject::connect(optixProc, &QProcess::finished, [=](int exitCode, QProcess::ExitStatus exitStatus) {

        //});

        //idle
        return app.exec();
    }

}
