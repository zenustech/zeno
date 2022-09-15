#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>

namespace {

    bool RtmpPush(std::string url) {
        // QStringList cmd = {"ffmpeg", "-y",    "-r",   QString::number(m_recordInfo.fps),           "-i",   path,
        //                "-c:v",   "mpeg4", "-b:v", QString::number(m_recordInfo.bitrate) + "k", outPath};

        // cmd = QString("ffmpeg -y -r %1 -i %2 -b:v %3k -c:v mpeg4 output.mp4")
        //           .arg(m_InitParam.iFps)
        //           .arg(m_InitParam.sPath + "/%07d.jpg")
        //           .arg(m_InitParam.iBitrate)
        //           .toStdString();
        // std::puts(cmd.c_str());
        // std::system(cmd.c_str());
    }

struct ZenoAR : zeno::INode{
    
    virtual void apply() override {
        auto outprim = new zeno::PrimitiveObject;

        auto sWebsocketIP = get_param<std::string>("WebsocketIP");
        auto sRtmpPushUrl = get_param<std::string>("RtmpPushUrl");    

        // bool ret = false;       

        // if(ret == false){
        //     zeno::log_error("CalcGeometryUV error");
        //     set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(new zeno::PrimitiveObject)));        
        //     return;
        // }
        // set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(outprim)));
    }
};

ZENDEFNODE(ZenoAR, 
{
    /*输入*/
    {},
    /*输出*/
    {
        "prim"
    },
    /*参数*/
    {
        {"string", "WebsocketIP", ""},
        {"string", "RtmpPushUrl", "rtmp://192.168.1.180:30001/live/myapp"},
    },
    /*类别*/
    {"math"}
});

}
