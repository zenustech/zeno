#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <filesystem>
#include <algorithm>
#include <array>
#include <fstream>
#include <cassert>
#include <zeno/types/UserData.h>
#include <unordered_set>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/types/PreparedFrame.h>
#include <chrono>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/document.h>
#include <zeno/types/IObjectXMacro.h>
#include <zeno/core/Session.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/string.h>
#include <zeno/utils/CppTimer.h>
#include <zeno/utils/fileio.h>
#include <zeno/extra/SceneAssembler.h>
#include <zeno/utils/Timer.h>

#ifdef __linux__
    #include<unistd.h>
    #include <sys/statfs.h>
#endif
#define MIN_DISKSPACE_MB 1024

#define _PER_OBJECT_TYPE(TypeName, ...) TypeName,
enum class ObjectType : int32_t {
    ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
};

namespace zeno {
struct GlobalComm::PlaybackCache {
    using Stamps=decltype(GlobalComm::m_inCacheFrames)::mapped_type;
    struct Token {
        std::array<std::pair<std::filesystem::file_time_type,uintmax_t>,7> files;
        bool valid=false;
        bool operator==(const Token& other) const {return valid && other.valid && files==other.files;}
    };
    struct Frame {
        ViewObjects objects;
        Stamps stamps;
        std::tuple<std::string,bool,std::string> runInfo;
        Token token;
        bool resident=false;
    };
    std::map<int,Frame> frames;
    PlaybackCacheStats stats;

    static Token token(const std::filesystem::path& root,int frame) {
        const auto dir=root/std::to_string(1000000+frame).substr(1);
        const std::array<std::filesystem::path,7> paths={dir/"stampInfo.txt",dir/"runInfo.txt",
            dir/"lightCameraObj.zencache",dir/"materialObj.zencache",dir/"matrixObj.zencache",
            dir/"normalObj.zencache",root/"data/stampInfo.txt"};
        Token result;
        for(size_t i=0;i<paths.size();++i) {
            std::error_code error;
            result.files[i].first=std::filesystem::last_write_time(paths[i],error);
            if(error)return result;
            result.files[i].second=std::filesystem::file_size(paths[i],error);
            if(error)return result;
        }
        result.valid=true;return result;
    }
};

ZENO_API GlobalComm::PlaybackCacheStats GlobalComm::playbackCacheStats() const {
    std::lock_guard lock(m_mtx);
    return m_playbackCache?m_playbackCache->stats:PlaybackCacheStats{};
}

ZENO_API void GlobalComm::preloadPlaybackFrames(size_t budgetBytes) {
    const auto start=std::chrono::steady_clock::now();
    std::string path;
    int begin=0,end=0;
    uint64_t generation=0;
    std::vector<int> completed;
    {
        std::lock_guard lock(m_mtx);
        if(cacheFramePath.empty() || !maxCachedFrames)return;
        if(!budgetBytes){m_playbackCache.reset();++m_cacheGeneration;return;}
        if(m_playbackCache && m_playbackCache->stats.budgetBytes==budgetBytes)return;
        if(m_playbackCache){m_playbackCache.reset();++m_cacheGeneration;}
        path=cacheFramePath;begin=beginFrameNumber;end=endFrameNumber;generation=m_cacheGeneration;
        for(size_t i=0;i<m_frames.size();++i)
            if(m_frames[i].frame_state==FRAME_COMPLETED)completed.push_back(begin+static_cast<int>(i));
    }
    if(completed.size()<3 || end<begin || completed.back()>end)return;
    auto prepared=std::make_shared<PlaybackCache>();prepared->stats.budgetBytes=budgetBytes;
    GlobalComm reader;reader.beginFrameNumber=begin;reader.endFrameNumber=end;
    reader.currentFrameNumber=begin-1;reader.m_frames.resize(size_t(end-begin+1));
    std::unordered_set<const IObject*> retained;
    std::array<std::shared_ptr<const nlohmann::json>,2> topologies;
    int previous=-1;
    for(const auto frame:completed) {
        const auto token=PlaybackCache::token(std::filesystem::u8path(path),frame);
        if(!token.valid)continue;
        const auto dir=std::filesystem::u8path(path)/std::to_string(1000000+frame).substr(1);
        const auto run=getRunType(dir);
        if(std::get<0>(run)!="RunAll" && std::get<0>(run)!="LoadAsset")continue;
        // Do not allocate an unexpectedly large frame merely to discover that
        // it cannot fit the bounded RAM cache. Ordinary playback remains valid.
        const auto largest=std::max({token.files[2].second,token.files[3].second,
                                    token.files[4].second,token.files[5].second});
        if(largest>budgetBytes/4)continue;
        const int index=frame-begin;
        auto& objects=reader.m_frames[index].view_objects;
        PlaybackCache::Stamps stamps;
        try {
            if(!reader.fromDiskByStampinfo(path,frame,objects,stamps,std::get<0>(run),true))continue;
        } catch(const std::exception&) {objects.clear();continue;}
        auto matrices=std::make_shared<PreparedMatrixStorage>();
        for(auto& [key,object]:objects) {
            auto* primitive=dynamic_cast<PrimitiveObject*>(object.get());
            if(!primitive)continue;
            const auto& ud=primitive->userData();
            const auto resource=ud.get2<std::string>("ResourceType","");
            if(resource=="Matrixes" && !dynamic_cast<PreparedMatrix*>(primitive) &&
                !ud.has("ShaderAttributes") && primitive->verts.size()%4==0) {
                auto matrix=std::make_shared<PreparedMatrix>();
                static_cast<PrimitiveObject&>(*matrix)=*primitive;
                matrix->matrixName=ud.get2<std::string>("ObjectName",key);
                matrix->unchanged=ud.get2<std::string>("stamp-change","TotalChange")=="UnChanged";
                matrix->offset=matrices->transforms.size();matrix->count=matrix->verts.size()/4;
                matrix->idOffset=matrices->instanceIds.size();matrix->idCount=matrix->loops.size();
                matrices->instanceIds.insert(matrices->instanceIds.end(),matrix->loops.values.begin(),matrix->loops.values.end());
                const auto* data=reinterpret_cast<const float*>(matrix->verts.data());
                for(size_t i=0;i<matrix->count;++i) {
                    std::array<float,12> transform;std::copy_n(data+i*12,12,transform.data());
                    glm::mat4 ui(1.0f);
                    for(size_t row=0;row<3;++row)for(size_t column=0;column<4;++column)ui[column][row]=transform[row*4+column];
                    matrices->transforms.push_back(transform);matrices->uiTransforms.push_back(ui);
                }
                matrix->storage=matrices;object=std::move(matrix);
            } else if(resource=="SceneTree" && !dynamic_cast<PreparedSceneTree*>(primitive)) {
                const auto type=ud.get2<std::string>("SceneTreeType","");
                if(type!="static" && type!="dynamic")continue;
                try {
                    auto topology=nlohmann::json::parse(ud.get2<std::string>("json"),
                        [](int depth,nlohmann::json::parse_event_t event,nlohmann::json& value) {
                            return !(depth==1 && event==nlohmann::json::parse_event_t::key &&
                                (value=="node_to_matrix" || value=="node_to_id"));
                        });
                    auto& shared=topologies[type=="static"?0:1];
                    if(!shared || *shared!=topology)shared=std::make_shared<const nlohmann::json>(std::move(topology));
                    auto tree=std::make_shared<PreparedSceneTree>();
                    static_cast<PrimitiveObject&>(*tree)=*primitive;tree->topology=shared;object=std::move(tree);
                } catch(const std::exception&) { /* Keep the ordinary decoder path. */ }
            }
        }
        size_t estimate=stamps.size()*512;
        for(const auto& [id,stamp]:stamps) {
            const auto object=objects.find(std::get<3>(stamp));
            if(object!=objects.end() && retained.insert(object->second.get()).second) {
                const size_t bytes=std::get<6>(stamp);
                if(bytes>budgetBytes/4){estimate=budgetBytes;break;}
                estimate+=bytes*4+512;
            }
        }
        if(!(token==PlaybackCache::token(std::filesystem::u8path(path),frame)))break;
        if(estimate>budgetBytes-std::min(budgetBytes,prepared->stats.estimatedBytes))break;
        PlaybackCache::Frame stored;stored.objects=objects;stored.stamps=stamps;stored.runInfo=run;stored.token=token;
        prepared->frames.emplace(frame,std::move(stored));prepared->stats.estimatedBytes+=estimate;
        reader.m_inCacheFrames.clear();reader.m_inCacheFrames.emplace(frame,std::move(stamps));reader.currentFrameNumber=frame;
        if(previous>=0)reader.m_frames[previous].view_objects.clear();previous=index;
    }
    prepared->stats.frames=prepared->frames.size();
    if(prepared->frames.empty())return;
    prepared->stats.preparationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    {
        std::lock_guard lock(m_mtx);
        if(generation!=m_cacheGeneration || path!=cacheFramePath || begin!=beginFrameNumber || end!=endFrameNumber)return;
        m_playbackCache=std::move(prepared);
    }
}

int secondLastColonIdx(const std::string& str) {
    int count = 0;
    for (int i = str.length() - 1; i >= 0; --i) {
        if (str[i] == ':') {
            ++count;
            if (count == 2)
                return i;
        }
    }
    return -1;
}

std::vector<std::filesystem::path> cachepath(4);
void GlobalComm::toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs, std::string runtype, bool balways, std::string fileName, bool isStampModeInit) {
    if (cachedir.empty()) return;
    m_playbackCache.reset();++m_cacheGeneration;

    std::filesystem::path dir = std::filesystem::u8path(cachedir + "/" + (isStampModeInit ? "data" : std::to_string(1000000 + frameid).substr(1)));
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir))
    {
        log_critical("can not create path: {}", dir);
    }

    auto otherNecessObjs = zeno::getSession().userData().get2<std::string>("otherNecessObjs", "");
    if (otherNecessObjs.empty()) {
        std::ifstream runinfoFile(dir / "runInfo.txt", std::ios::binary);
        if (runinfoFile.is_open()) {
            rapidjson::Document doc;
            std::stringstream buffer;
            buffer << runinfoFile.rdbuf();
            doc.Parse(buffer.str().c_str());
            auto obj = doc.GetObject();
            if (obj.HasMember("otherNecessObjs")) {
                otherNecessObjs = obj["otherNecessObjs"].GetString();
            }
        }
    }

    rapidjson::StringBuffer runinfoStrBuf;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> runinfowriter(runinfoStrBuf);
    runinfowriter.StartObject();
    runinfowriter.Key("runtype");
    runinfowriter.String(runtype.c_str());
    runinfowriter.Key("always");
    runinfowriter.Bool(balways);
    runinfowriter.Key("viewNodes");
    runinfowriter.String(allViewNodes.c_str());
    runinfowriter.Key("otherNecessObjs");
    runinfowriter.String(otherNecessObjs.c_str());
    runinfowriter.EndObject();
    std::string strRuninfo = runinfoStrBuf.GetString();
    std::filesystem::path runInfoPath = dir / "runInfo.txt";
    std::ofstream runinfoofs(runInfoPath, std::ios::binary);
    std::ostreambuf_iterator<char> runinfooit(runinfoofs);
    std::copy(strRuninfo.begin(), strRuninfo.end(), runinfooit);

    std::vector<std::vector<char>> bufCaches(4);
    std::vector<std::vector<size_t>> poses(4);
    std::vector<std::string> keys(4);

    std::filesystem::path stampInfoPath = dir / "stampInfo.txt";
    bool hasStampNode = zeno::getSession().userData().has("graphHasStampNode") || std::filesystem::exists(stampInfoPath);
    if (hasStampNode) {
        std::filesystem::path stampInfoPath = dir / "stampInfo.txt";
        std::map<std::string, std::tuple<std::string, int>> lastframeStampinfo;

        if (!isStampModeInit && frameid != beginFrameNumber) {
            rapidjson::Document doc;
            std::filesystem::path lastframeStampPath = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid - 1).substr(1)) / "stampInfo.txt";
            std::ifstream file(lastframeStampPath);
            if (file) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                doc.Parse(buffer.str().c_str());
                if (doc.IsObject()) {
                    for (const auto& group : doc.GetObject()) {
                        if (group.value.IsObject()) {
                            for (const auto& node : group.value.GetObject()) {
                                const std::string& key = node.name.GetString();
                                lastframeStampinfo.insert({ key.substr(0, secondLastColonIdx(key)), std::tuple<std::string, int>(node.value["stamp-change"].GetString(), node.value["stamp-base"].GetInt()) });
                            }
                        }
                    }
                }
            }
        }

        rapidjson::StringBuffer lightCameraStr, materialStr, matrixStr, normalStr;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> lightCamerawriter(lightCameraStr), materialwriter(materialStr), matrixwriter(matrixStr), normalwriter(normalStr);
        lightCamerawriter.StartObject();
        lightCamerawriter.Key("lightCameraObj");
        lightCamerawriter.StartObject();
        materialwriter.StartObject();
        materialwriter.Key("materialObj");
        materialwriter.StartObject();
        matrixwriter.StartObject();
        matrixwriter.Key("matrixObj");
        matrixwriter.StartObject();
        normalwriter.StartObject();
        normalwriter.Key("normalObj");
        normalwriter.StartObject();
        for (auto& [key, obj] : objs) {
            /*if (isBeginframe) {
                obj->userData().set2("stamp-change", "TotalChange");
            }*/
            std::string oldStampChange = obj->userData().get2<std::string>("stamp-change", "TotalChange");

            std::string stamptag = isStampModeInit ? "TotalChange" : oldStampChange;
            const auto previousStamp = lastframeStampinfo.find(key.substr(0, secondLastColonIdx(key)));
            int baseframe = isStampModeInit ? -9999 :
                (stamptag == "TotalChange" || frameid == beginFrameNumber || previousStamp == lastframeStampinfo.end()
                    ? frameid : std::get<1>(previousStamp->second));
            obj->userData().set2("stamp-base", baseframe);
            obj->userData().set2("stamp-change", stamptag);


            //写出stampinfo
            const auto& objRunType = obj->userData().get2<std::string>("objRunType", "normal");
            auto idx = objRunType == "lightCamera" ? 0 : (objRunType == "material" ? 1 : (objRunType == "matrix" ? 2 : 3));

            rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer = idx == 0 ? lightCamerawriter : (idx == 1 ? materialwriter : (idx == 2 ? matrixwriter : normalwriter));
            writer.Key(key.c_str());
            writer.StartObject();
            writer.Key("stamp-change");
            writer.String(stamptag.c_str());
            writer.Key("stamp-base");
            writer.Int(baseframe);

            writer.Key("objRunType");
            writer.Int(idx);

            if (0) {
#define _PER_OBJECT_TYPE(TypeName, ...) \
            } else if (auto o = dynamic_cast<TypeName const *>(obj.get())) { \
                writer.Key("stamp-objType"); \
                writer.Int((int)ObjectType::TypeName);
                ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
#undef _PER_OBJECT_TYPE
            }
            else {
                writer.Key("objType");
                writer.Int(-1);
            }
            if (stamptag == "DataChange") {
                writer.Key("stamp-dataChange-hint");
                std::string changehint = obj->userData().get2<std::string>("stamp-dataChange-hint", "");
                writer.String(changehint.c_str());
            }

            //编码obj
            if (stamptag == "UnChanged") {
                //如果是计算第一帧的lightcamera，material，matrix，输出cache，如果是always运行的lightcamera，material，matrix也输出cache(loadasset全是totalchange不需要考虑)
                //if (frameid == beginFrameNumber && runtype != "LoadAsset" && runtype != "RunAll" ||
                //    balways && runtype != "LoadAsset" && runtype != "RunAll") {
                //} else {
					//writer.EndObject();
					//continue;
					std::shared_ptr<IObject> unchangeObj;
					if (0) {
#define _PER_OBJECT_TYPE(TypeName, ...) \
                    } else if (auto o = dynamic_cast<TypeName const *>(obj.get())) { \
                        unchangeObj = constructEmptyObj((int)ObjectType::TypeName);
						ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
#undef _PER_OBJECT_TYPE
					}
					else {
						auto unchangeObj = constructEmptyObj(0);
					}
                    // View objects may also be retained by cached graph nodes.
                    unchangeObj->m_userData = obj->userData();
                    obj = unchangeObj;
                //}
            }
            else if (stamptag == "DataChange") {
                int baseframe = obj->userData().get2<int>("stamp-base", -1);
                std::string changehint = obj->userData().get2<std::string>("stamp-dataChange-hint", "");
                //TODO:
                //data = obj.根据changehint获取变化的data
                if (0) {
#define _PER_OBJECT_TYPE(TypeName, ...) \
            } else if (auto o = std::dynamic_pointer_cast<TypeName>(obj)) { \
                obj = std::make_shared<zeno::TypeName>();   //置为空obj
                    ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
#undef _PER_OBJECT_TYPE
                }
            else {
                }
                obj->userData().set2("stamp-change", "DataChange");
                obj->userData().set2("stamp-base", baseframe);
                obj->userData().set2("stamp-dataChange-hint", changehint);
                //TODO:
                //obj根据changehint设置data更新的部分
            }
            else if (stamptag == "ShapeChange") {
                int baseframe = obj->userData().get2<int>("stamp-base", -1);
                //暂时并入Totalchange:
                //shape = obj.获取shape()
                if (0) {
#define _PER_OBJECT_TYPE(TypeName, ...) \
            } else if (auto o = std::dynamic_pointer_cast<TypeName>(obj)) { \
                obj = std::make_shared<zeno::TypeName>();   //置为空obj
                    ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
#undef _PER_OBJECT_TYPE
                }
            else {
                }
                obj->userData().set2("stamp-change", "ShapeChange");
                obj->userData().set2("stamp-base", baseframe);
                //TODO:
                //obj.设置shape更新的部分
            }
            size_t bufsize = 0;
            bufsize = bufCaches[idx].size();
            if (encodeObject(obj.get(), bufCaches[idx]))
            {
                keys[idx].push_back('\a');
                keys[idx].append(key);
                poses[idx].push_back(bufsize);
            }
            if (isStampModeInit) {
                obj->userData().set2("stamp-change", oldStampChange);
            }

            //写出stampinfo的obj尺寸
            writer.Key("startIndexInCache");
            writer.String(std::to_string(bufsize).c_str());
            writer.Key("ObjSize");
            writer.String(std::to_string(bufCaches[idx].size() - bufsize).c_str());
            writer.EndObject();
        }
        //写出stampinfo
        lightCamerawriter.EndObject();
        lightCamerawriter.EndObject();
        materialwriter.EndObject();
        materialwriter.EndObject();
        matrixwriter.EndObject();
        matrixwriter.EndObject();
        normalwriter.EndObject();
        normalwriter.EndObject();
        if (runtype != "RunAll" && runtype != "LoadAsset") {
            rapidjson::Document currFrameStampInfodoc;
            std::filesystem::path lastframeStampPath = std::filesystem::u8path(cachedir + "/" + (isStampModeInit ? "data" : std::to_string(1000000 + frameid).substr(1))) / "stampInfo.txt";
            std::ifstream currFrameiffile(lastframeStampPath);
            std::stringstream currFramestampinfobuffer;
            currFramestampinfobuffer << currFrameiffile.rdbuf();
            currFrameStampInfodoc.Parse(currFramestampinfobuffer.str().c_str());

            std::string objtype = runtype == "RunLightCamera" ? "lightCameraObj" : (runtype == "RunMaterial" ? "materialObj" : (runtype == "RunMatrix" ? "matrixObj" : "normalObj"));
            if (currFrameStampInfodoc.IsObject()) {
                for (auto& member : currFrameStampInfodoc.GetObject()) {
                    std::string key = std::string(member.name.GetString());
                    if (key != objtype) {
                        const rapidjson::Value& value = member.value;
                        if (key == "lightCameraObj") {
                            lightCameraStr.Clear();
                            lightCamerawriter.Reset(lightCameraStr);
                            lightCamerawriter.StartObject();
                            lightCamerawriter.Key(key.c_str());
                            value.Accept(lightCamerawriter);
                            lightCamerawriter.EndObject();
                        }
                        else if (key == "materialObj") {
                            materialStr.Clear();
                            materialwriter.Reset(materialStr);
                            materialwriter.StartObject();
                            materialwriter.Key(key.c_str());
                            value.Accept(materialwriter);
                            materialwriter.EndObject();
                        }
                        else if (key == "matrixObj") {
                            matrixStr.Clear();
                            matrixwriter.Reset(matrixStr);
                            matrixwriter.StartObject();
                            matrixwriter.Key(key.c_str());
                            value.Accept(matrixwriter);
                            matrixwriter.EndObject();
                        }
                        else {
                            normalStr.Clear();
                            normalwriter.Reset(normalStr);
                            normalwriter.StartObject();
                            normalwriter.Key(key.c_str());
                            value.Accept(normalwriter);
                            normalwriter.EndObject();
                        }
                    }
                }
            }
        }
        const auto& removeParentheses = [](std::string& str) {
            str = str.substr(str.find_first_of('{') + 1);
            str = str.substr(0, str.find_last_of('}'));
        };
        std::string lightcamerajson = lightCameraStr.GetString(), materialjson = materialStr.GetString(), matrixjson = matrixStr.GetString(), normaljson = normalStr.GetString();
        removeParentheses(lightcamerajson);
        removeParentheses(materialjson);
        removeParentheses(matrixjson);
        removeParentheses(normaljson);
        std::string res = "{" + lightcamerajson + "," + materialjson + "," + matrixjson + "," + normaljson + "}";
        std::ofstream ofs(stampInfoPath, std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(res.begin(), res.end(), oit);
    }
    else {
        for (auto& [key, obj] : objs) {
            const auto& objRunType = obj->userData().get2<std::string>("objRunType", "normal");
            auto idx = objRunType == "lightCamera" ? 0 : (objRunType == "material" ? 1 : (objRunType == "matrix" ? 2 : 3));
            size_t bufsize = 0;
            bufsize = bufCaches[idx].size();
            if (encodeObject(obj.get(), bufCaches[idx]))
            {
                keys[idx].push_back('\a');
                keys[idx].append(key);
                poses[idx].push_back(bufsize);
            }
        }
    }

    cachepath[0] = dir / "lightCameraObj.zencache";
    cachepath[1] = dir / "materialObj.zencache";
    cachepath[2] = dir / "matrixObj.zencache";
    cachepath[3] = dir / "normalObj.zencache";

    size_t currentFrameSize = 0;

    for (int i = 0; i < 4; i++)
    {
        //if (poses[i].size() == 0 && (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2))
        if (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2)
            continue;
        keys[i].push_back('\a');
        keys[i] = "ZENCACHE" + std::to_string(poses[i].size()) + keys[i];
        poses[i].push_back(bufCaches[i].size());
        currentFrameSize += keys[i].size() + poses[i].size() * sizeof(size_t) + bufCaches[i].size();
    }

    size_t freeSpace = 0;
    #ifdef __linux__
        struct statfs diskInfo;
        statfs(std::filesystem::u8path(cachedir).c_str(), &diskInfo);
        freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;
    #else
        freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
    #endif
    //wait in two case: 1. available space minus current frame size less than 1024MB, 2. available space less or equal than 1024MB
    while ( ((freeSpace >> 20) - MIN_DISKSPACE_MB) < (currentFrameSize >> 20)  || (freeSpace >> 20) <= MIN_DISKSPACE_MB)
    {
        #ifdef __linux__
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).string());
            sleep(2);
            statfs(std::filesystem::u8path(cachedir).c_str(), &diskInfo);
            freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;

        #else
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).root_path().string());
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
        #endif
    }

    for (int i = 0; i < 4; i++)
    {
        //if (poses[i].size() == 0 && (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2))
        if (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2)
            continue;
        log_debug("dump cache to disk {}", cachepath[i]);
        std::ofstream ofs(cachepath[i], std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(keys[i].begin(), keys[i].end(), oit);
        std::copy_n((const char*)poses[i].data(), poses[i].size() * sizeof(size_t), oit);
        std::copy(bufCaches[i].begin(), bufCaches[i].end(), oit);
    }

    if (!isStampModeInit) {
        objs.clear();
    }
}

bool GlobalComm::fromDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs, std::string& runtype, std::string fileName) {
    if (cachedir.empty())
        return false;
    objs.clear();

    auto dir = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1);

    //runtype = getRunType(dir);
    //if (currentFrameNumber != frameid) {//切帧要加载新帧的全部
    //    runtype = "RunAll";
    //}

    cachepath[0] = dir / "lightCameraObj.zencache";
    cachepath[1] = dir / "materialObj.zencache";
    cachepath[2] = dir / "matrixObj.zencache";
    cachepath[3] = dir / "normalObj.zencache";
    if (runtype == "RunLightCamera") {
    } else if (runtype == "RunMaterial") {
        std::swap(cachepath[0], cachepath[1]);
    } else if (runtype == "RunMatrix") {
        std::swap(cachepath[0], cachepath[2]);
    }

    std::set<std::string> objAdded;

    for (int i = 0; i < cachepath.size(); i++)
    {
        //if (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2) {
        //    continue;
        //}
        if (!std::filesystem::exists(cachepath[i]))
        {
            continue;
        }
        log_debug("load cache from disk {}", cachepath[i]);

        auto szBuffer = std::filesystem::file_size(cachepath[i]);
        std::vector<char> dat(szBuffer);
        FILE* fp = fopen(cachepath[i].string().c_str(), "rb");
        if (!fp) {
            log_error("zeno cache file does not exist");
            return false;
        }
        size_t ret = fread(&dat[0], 1, szBuffer, fp);
        assert(ret == szBuffer);
        fclose(fp);
        fp = nullptr;

        if (dat.size() <= 8 || std::string(dat.data(), 8) != "ZENCACHE") {
            log_error("zeno cache file broken (1)");
            return false;
        }
        size_t pos = std::find(dat.begin() + 8, dat.end(), '\a') - dat.begin();
        if (pos == dat.size()) {
            log_error("zeno cache file broken (2)");
            return false;
        }
        size_t keyscount = std::stoi(std::string(dat.data() + 8, pos - 8));
        pos = pos + 1;
        std::vector<std::string> keys;
        for (int k = 0; k < keyscount; k++) {
            size_t newpos = std::find(dat.begin() + pos, dat.end(), '\a') - dat.begin();
            if (newpos == dat.size()) {
                log_error("zeno cache file broken (3.{})", k);
                return false;
            }
            keys.emplace_back(dat.data() + pos, newpos - pos);
            pos = newpos + 1;
        }
        std::vector<size_t> poses(keyscount + 1);
        std::copy_n(dat.data() + pos, (keyscount + 1) * sizeof(size_t), (char*)poses.data());
        pos += (keyscount + 1) * sizeof(size_t);
        for (int k = 0; k < keyscount; k++) {
            if (poses[k] > dat.size() - pos || poses[k + 1] < poses[k]) {
                log_error("zeno cache file broken (4.{})", k);
            }
            const char* p = dat.data() + pos + poses[k];

            if (runtype != "RunAll" && runtype != "LoadAsset") {
                auto decodedObj = decodeObject(p, poses[k + 1] - poses[k]);
                if (objAdded.count(keys[k].substr(0, secondLastColonIdx(keys[k]))) == 0) {
                    objs.try_emplace(keys[k], decodedObj);
                    objAdded.insert(keys[k].substr(0, secondLastColonIdx(keys[k])));
                }
            } else {
                objs.try_emplace(keys[k], decodeObject(p, poses[k + 1] - poses[k]));
            }
        }
    }
    return true;
}

int GlobalComm::getObjType(std::shared_ptr<IObject> obj)
{
    if (0) {
#define _PER_OBJECT_TYPE(TypeName, ...) \
    } else if (auto o = std::dynamic_pointer_cast<TypeName>(obj)) { \
        return (int)ObjectType::TypeName;
        ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
#undef _PER_OBJECT_TYPE
    } else {
    }
}

std::shared_ptr<zeno::IObject> GlobalComm::constructEmptyObj(int type)
{
    if (0) {
#define _PER_OBJECT_TYPE(TypeName, ...) \
    } else if ((int)ObjectType::TypeName == type) { \
        return std::make_shared<zeno::TypeName>();
        ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
#undef _PER_OBJECT_TYPE
    } else {
    }
}

bool GlobalComm::fromDiskByStampinfo(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>>& newFrameStampInfo, std::string runtype, bool switchTimeline, bool loadasset, std::string currViewNodes)
{
    if (cachedir.empty()) return false;
    using StampInfo = std::decay_t<decltype(newFrameStampInfo)>;
    using StampIterator = StampInfo::iterator;
    const auto root = std::filesystem::u8path(cachedir);
    const auto dir = root / (loadasset ? "data" : std::to_string(1000000 + frameid).substr(1));
    const std::array<const char*, 4> categories = {"lightCameraObj", "materialObj", "matrixObj", "normalObj"};
    const int runCategory = runtype == "RunLightCamera" ? 0 : runtype == "RunMaterial" ? 1 : runtype == "RunMatrix" ? 2 : -1;
    std::array<std::filesystem::path, 4> paths;
    for (int i = 0; i < 4; ++i) {
        const bool fromAssets = !switchTimeline && !loadasset && i < 3 &&
            (runtype == "RunAll" || i == runCategory);
        paths[i] = (fromAssets ? root / "data" : dir) / (std::string(categories[i]) + ".zencache");
    }

    const auto readStamps = [](const std::filesystem::path& path, rapidjson::Document& doc) {
        std::ifstream file(path);
        if (!file) return false;
        std::stringstream buffer;
        buffer << file.rdbuf();
        doc.Parse(buffer.str().c_str());
        return !doc.HasParseError() && doc.IsObject();
    };
    rapidjson::Document doc, assetDoc;
    if (!readStamps(dir / "stampInfo.txt", doc)) return false;
    if (!switchTimeline && !loadasset && (runtype == "RunAll" || runCategory >= 0) &&
        !readStamps(root / "data/stampInfo.txt", assetDoc)) return false;

    const StampInfo* previousStamps = nullptr;
    const ViewObjects* previousObjects = nullptr;
    const auto previous = m_inCacheFrames.find(currentFrameNumber);
    const int previousIndex = currentFrameNumber - beginFrameNumber;
    // A rerun or asset reload must read fresh payloads, even when node names match.
    if (switchTimeline && !loadasset && frameid != currentFrameNumber &&
        previous != m_inCacheFrames.end() && previousIndex >= 0 && previousIndex < m_frames.size()) {
        previousStamps = &previous->second;
        previousObjects = &m_frames[previousIndex].view_objects;
    }
    std::array<std::vector<StampIterator>, 4> required;
    std::array<int, 4> order = {0, 1, 2, 3};
    if (runCategory >= 0) std::swap(order[0], order[runCategory]);
    for (const int category : order) {
        const bool fromAssets = !switchTimeline && !loadasset && category < 3 &&
            (runtype == "RunAll" || category == runCategory);
        const auto& source = fromAssets ? assetDoc : doc;
        if (!source.HasMember(categories[category]) || !source[categories[category]].IsObject()) return false;
        for (const auto& node : source[categories[category]].GetObject()) {
            const std::string key = node.name.GetString();
            const auto nodeid = key.substr(0, secondLastColonIdx(key));
            if (runtype != "RunAll" && runtype != "LoadAsset" &&
                currViewNodes.find(nodeid.substr(0, nodeid.find_first_of(':'))) == std::string::npos) continue;
            if (!node.value.IsObject()) return false;
            const auto& value = node.value;
            const std::string change = value.HasMember("stamp-change") ? value["stamp-change"].GetString() : "TotalChange";
            const int base = value.HasMember("stamp-base") ? value["stamp-base"].GetInt() : -1;
            const int type = value.HasMember("stamp-objType") ? value["stamp-objType"].GetInt() : 0;
            const std::string hint = value.HasMember("stamp-dataChange-hint") ? value["stamp-dataChange-hint"].GetString() : "";
            const size_t offset = value.HasMember("startIndexInCache") ? std::stoull(value["startIndexInCache"].GetString()) : 0;
            const size_t length = value.HasMember("ObjSize") ? std::stoull(value["ObjSize"].GetString()) : 0;
            const auto [stamp, inserted] = newFrameStampInfo.emplace(nodeid,
                std::make_tuple(change, base, type, key, hint, offset, length));
            if (!inserted) continue;

            if (change == "UnChanged" && previousStamps) {
                const auto old = previousStamps->find(nodeid);
                if (old != previousStamps->end() && std::get<0>(old->second) == "UnChanged" &&
                    std::get<1>(old->second) == base && std::get<2>(old->second) == type) {
                    const auto& oldKey = std::get<3>(old->second);
                    const auto object = previousObjects->find(oldKey);
                    if (object != previousObjects->end() && object->second &&
                        object->second->userData().get2<std::string>("stamp-change", "TotalChange") == "UnChanged") {
                        // Lights are re-enumerated on each frame by ObjectsManager.
                        // Retain their payload, but preserve the incoming registration key.
                        auto& resolvedKey = std::get<3>(stamp->second);
                        if (category != 0 && !object->second->userData().get2<int>("isRealTimeObject", 0)) {
                            resolvedKey = oldKey;
                        }
                        objs.try_emplace(resolvedKey, object->second);
                        continue;
                    }
                }
            }
            if (change == "TotalChange" || change == "UnChanged") required[category].push_back(stamp);
        }
    }

    for (int category = 0; category < 4; ++category) {
        auto& requests = required[category];
        if (requests.empty()) continue;
        std::ifstream file(paths[category], std::ios::binary);
        if (!file) return false;
        const auto fileSize = std::filesystem::file_size(paths[category]);
        std::string header;
        if (!std::getline(file, header, '\a') || header.compare(0, 8, "ZENCACHE") != 0) return false;
        const size_t keyCount = std::stoull(header.substr(8));
        if (keyCount > fileSize / sizeof(size_t)) return false;
        for (size_t i = 0; i < keyCount; ++i) {
            if (!std::getline(file, header, '\a')) return false;
        }
        file.seekg((keyCount + 1) * sizeof(size_t), std::ios::cur);
        const auto position = file.tellg();
        if (position < 0 || static_cast<size_t>(position) > fileSize) return false;
        const size_t payload = static_cast<size_t>(position);
        for (const auto stamp : requests) {
            const auto offset = std::get<5>(stamp->second), length = std::get<6>(stamp->second);
            if (!length || offset > fileSize - payload || length > fileSize - payload - offset) return false;
        }
        std::sort(requests.begin(), requests.end(), [](const auto left, const auto right) {
            return std::get<5>(left->second) < std::get<5>(right->second);
        });
        for (size_t first = 0; first < requests.size();) {
            const size_t offset = std::get<5>(requests[first]->second);
            size_t end = offset + std::get<6>(requests[first]->second), last = first + 1;
            // Coalesce adjacent required payloads without rereading skipped static data.
            while (last < requests.size() && std::get<5>(requests[last]->second) == end &&
                std::get<6>(requests[last]->second) <= 8 * 1024 * 1024 - std::min(end - offset, size_t(8 * 1024 * 1024))) {
                end += std::get<6>(requests[last]->second);
                ++last;
            }
            std::vector<char> buffer(end - offset);
            file.seekg(payload + offset, std::ios::beg);
            if (!file.read(buffer.data(), buffer.size())) return false;
            for (size_t i = first; i < last; ++i) {
                const auto& stamp = requests[i]->second;
                auto object = decodeObject(buffer.data() + std::get<5>(stamp) - offset, std::get<6>(stamp));
                if (!object) return false;
                const auto& resource = !switchTimeline
                    ? object->userData().get2<std::string>("ResourceType", "") : std::string();
                const bool forceRegistration = !switchTimeline &&
                    (category == runCategory || resource == "SceneDescriptor" || resource == "SceneTree");
                if (forceRegistration && std::get<0>(stamp) == "UnChanged") {
                    object->userData().set2("stamp-change", "TotalChange");
                } else if (std::get<0>(stamp) == "UnChanged") {
                    object->userData().set2("stamp-change", "UnChanged");
                }
                objs.try_emplace(std::get<3>(stamp), std::move(object));
            }
            first = last;
        }
    }
    return true;
}

std::shared_ptr<IObject> GlobalComm::fromDiskReadObject(std::string cachedir, int frameid, std::string objectName)
{
    if (cachedir.empty())
        return nullptr;
    auto dir = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1);

    std::string runtype = std::get<0>(getRunType(dir));

    cachepath[0] = dir / "lightCameraObj.zencache";
    cachepath[1] = dir / "materialObj.zencache";
    cachepath[2] = dir / "matrixObj.zencache";
    cachepath[3] = dir / "normalObj.zencache";

    for (int i = 0; i < cachepath.size(); i++)
    {
        if (!std::filesystem::exists(cachepath[i]))
        {
            continue;
        }
        log_debug("load cache from disk {}", cachepath[i]);

        auto szBuffer = std::filesystem::file_size(cachepath[i]);

        std::ifstream file(cachepath[i], std::ios::binary);
        if (!file.is_open()) {
            log_error("zeno cache file does not exist");
            return nullptr;
        }
        std::string keysStr;
        std::getline(file, keysStr, '\a');
        size_t pos = keysStr.size();
        if (pos <= 8) {
            log_error("zeno cache file broken");
            return nullptr;
        }

        size_t keyscount = std::stoi(keysStr.substr(8));
        if (keyscount < 1) {
            continue;
        }
        size_t keyindex = -1;
        std::string targetkey;
        for (int k = 0; k < keyscount; k++) {
            std::string segment;
            std::getline(file, segment, '\a');
            if (segment.find(objectName) != std::string::npos) {
                keyindex = k;
                targetkey = segment;
            }
        }
        if (keyindex == -1) {
            continue;
        }
        else {
            std::vector<size_t> poses(keyscount + 1);
            file.read((char*)poses.data(), (keyscount + 1) * sizeof(size_t));
            size_t posstart = poses[keyindex], posend = poses[keyindex + 1];

            if (posend < posstart || szBuffer < posend) {
                log_error("zeno cache file broken");
                return nullptr;
            }
            std::vector<char> objbuff(posend - posstart);
            file.seekg(posstart, std::ios::cur);
            file.read(objbuff.data(), posend - posstart);
            return decodeObject(objbuff.data(), posend - posstart);
        }
    }
    return nullptr;
}

std::tuple<std::string, bool, std::string> GlobalComm::getRunType(std::filesystem::path dir)
{
    std::filesystem::path runinfoPath = dir / "runInfo.txt";
    std::ifstream runinfoFile(runinfoPath, std::ios::binary);
    if (!runinfoFile.is_open()) {
        log_error("run info file does not exist");
        return std::make_tuple("RunAll", false, "");
    }
    rapidjson::Document doc;
    std::stringstream buffer;
    buffer << runinfoFile.rdbuf();
    doc.Parse(buffer.str().c_str());

    std::tuple<std::string, bool, std::string> runinfo("RunAll", false, "");
    if (doc.IsObject()) {
        auto obj = doc.GetObject();
        if (obj.HasMember("runtype")) {
            std::get<0>(runinfo) = obj["runtype"].GetString();
        }
        if (obj.HasMember("always")) {
            std::get<1>(runinfo) = obj["always"].GetBool();
        }
        if (obj.HasMember("viewNodes")) {
            std::string otherNecessObjs;
            if (obj.HasMember("otherNecessObjs")) {
                otherNecessObjs = obj["otherNecessObjs"].GetString();
            }
            std::get<2>(runinfo) = obj["viewNodes"].GetString() + otherNecessObjs;
        }
    }
    return runinfo;
}

ZENO_API void GlobalComm::newFrame() {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::newFrame {}", m_frames.size());
    m_frames.emplace_back().frame_state = FRAME_UNFINISH;
}

ZENO_API void GlobalComm::finishFrame() {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::finishFrame {}", m_maxPlayFrame);
    if (m_maxPlayFrame >= 0 && m_maxPlayFrame < m_frames.size())
        m_frames[m_maxPlayFrame].frame_state = FRAME_COMPLETED;
    m_maxPlayFrame += 1;
}

ZENO_API void GlobalComm::dumpFrameCache(int frameid, std::string runtype, bool balways) {
    std::lock_guard lck(m_mtx);
    auto get_format = [] (std::string const &key) -> std::string {
        auto prefix = key.substr(0, key.find(":LIST")+5);
        std::string count_str = key.substr(key.find(":LIST")+5);
        auto postfix = count_str.substr(count_str.find(':'));
        return prefix+"{}"+postfix;
    };
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx >= 0 && frameIdx < m_frames.size()) {
        {
            std::string dynamic_json_key;
            std::string dynamic_prefix;
            std::string static_json_key;
            std::string static_prefix;
            for (auto &[key, obj]: m_frames[frameIdx].view_objects.m_curr) {
                auto ResourceType = obj->userData().get2("ResourceType", std::string(""));
                if (ResourceType == "SceneTree") {
                    auto scene_tree = obj->userData().get2<std::string>("json");
                    if (scene_tree == static_scene_tree) {
                        continue;
                    }
                    Json json = Json::parse(scene_tree);
                    if (json["type"] == "dynamic") {
                        dynamic_json_key = key;
                        dynamic_prefix = key.substr(0, key.find(":LIST"));
                    }
                    else if (json["type"] == "static") {
                        static_json_key = key;
                        static_prefix = key.substr(0, key.find(":LIST"));
                        static_scene_tree = scene_tree;
                    }
                }
            }
            ViewObjects view_objects;
            auto dynamic_list = std::make_shared<ListObject>();
            auto static_list = std::make_shared<ListObject>();
            for (auto &[key, obj]: m_frames[frameIdx].view_objects.m_curr) {
                if (dynamic_prefix.size() > 0 && zeno::starts_with(key, dynamic_prefix)) {
                    dynamic_list->arr.push_back(obj);
                }
                else if (static_prefix.size() > 0 && zeno::starts_with(key, static_prefix)) {
                    static_list->arr.push_back(obj);
                }
                else {
                    view_objects.m_curr[key] = obj;
                }
            }
            if (dynamic_json_key.size()) {
                auto pattern = get_format(dynamic_json_key);
                auto scene = get_scene_tree_from_list2(dynamic_list);
                auto new_list = scene->to_structure();
                auto count = new_list->arr.size();
                auto sd = new_list->arr[count - 2];
                auto json_str = sd->userData().get2<std::string>("Scene", "");
                if (!json_str.empty()) {
                    dynamic_scene_descriptor = Json::parse(json_str);
                }
                sd->userData().set2("ResourceType", std::string("DeletedSceneDescriptor"));
                for (auto i = 0; i < new_list->arr.size(); i++) {
                    view_objects.m_curr[zeno::format(pattern, i)] = new_list->arr[i];
                }
            }
            if (frameid == beginFrameNumber && runtype == "LoadAsset" && static_json_key.size() > 0) {
                auto pattern = get_format(static_json_key);
                auto scene = get_scene_tree_from_list2(static_list);
                auto new_list = scene->to_structure();
                auto count = new_list->arr.size();
                auto sd = new_list->arr[count - 2];
                auto json_str = sd->userData().get2<std::string>("Scene", "");
                if (!json_str.empty()) {
                    static_scene_descriptor = Json::parse(json_str);
                }
                sd->userData().set2("ResourceType", std::string("DeletedSceneDescriptor"));
                for (auto i = 0; i < new_list->arr.size(); i++) {
                    view_objects.m_curr[zeno::format(pattern, i)] = new_list->arr[i];
                }
            }
            {
                std::unordered_map<std::string, std::vector<std::string>> matrixes;
                for (const auto&[key, obj]: view_objects.m_curr) {
                    if (obj->userData().get2<std::string>("ResourceType", "") == "Matrixes") {
                        auto obj_name = obj->userData().get2<std::string>("ObjectName", "");
                        if (obj_name == "") {
                            continue;
                        }
                        matrixes[obj_name].push_back(key);
                    }
                }
                for (auto &[obj_name, keys]: matrixes) {
                    if (keys.size() <= 1) {
                        continue;
                    }
                    std::string max_priority_obj_name = keys[0];
                    int max_priority = view_objects.m_curr[keys[0]]->userData().get2<int>("MatrixPriority", 0);
                    for (auto i = 1; i < keys.size(); i++) {
                        auto key = keys[i];
                        int priority = view_objects.m_curr[keys[i]]->userData().get2<int>("MatrixPriority", 0);
                        if (priority > max_priority) {
                            max_priority = priority;
                            max_priority_obj_name = key;
                        }
                    }
                    for (auto i = 0; i < keys.size(); i++) {
                        auto key = keys[i];
                        if (key != max_priority_obj_name) {
                            view_objects.m_curr[keys[i]]->userData().set2<std::string>("ResourceType", "DeleteMatrix");
                        }
                    }
                }
            }
            m_frames[frameIdx].view_objects = view_objects;
        }
        {
            Json scene_descriptor_json;
            if (!static_scene_descriptor.is_null()) {
                auto &json = static_scene_descriptor;
                if (!json["BasicRenderInstances"].is_null()) {
                    scene_descriptor_json["BasicRenderInstances"].update(json["BasicRenderInstances"]);
                }
                if (json.contains("StaticRenderGroups") && !json["StaticRenderGroups"].is_null()) {
                    scene_descriptor_json["StaticRenderGroups"].update(json["StaticRenderGroups"]);
                }
                if (json.contains("StaticEntries") && !json["StaticEntries"].is_null()) {
                    scene_descriptor_json["StaticEntries"].update(json["StaticEntries"]);
                }
            }
            if (!dynamic_scene_descriptor.is_null()) {
                auto &json = dynamic_scene_descriptor;
                if (!json["BasicRenderInstances"].is_null()) {
                    scene_descriptor_json["BasicRenderInstances"].update(json["BasicRenderInstances"]);
                }
                if (json.contains("DynamicRenderGroups") && !json["DynamicRenderGroups"].is_null()) {
                    scene_descriptor_json["DynamicRenderGroups"].update(json["DynamicRenderGroups"]);
                }

                if (json.contains("DynamicEntries") && !json["DynamicEntries"].is_null()) {
                    scene_descriptor_json["DynamicEntries"].update(json["DynamicEntries"]);
                }
            }
            if (scene_descriptor_json.empty() == false) {
                auto scene_descriptor = std::make_shared<zeno::PrimitiveObject>();
                auto &ud = scene_descriptor->userData();
                ud.set2("ResourceType", std::string("SceneDescriptor"));
                ud.set2("Scene", std::string(scene_descriptor_json.dump()));
                std::srand(std::time(0));
                auto json_key = zeno::format("GeneratedJson:{}", std::rand());
    //            zeno::file_put_content("E:/fuck/Generated.json", ud.get2<std::string>("Scene"));
                m_frames[frameIdx].view_objects.m_curr[json_key] = scene_descriptor;

                //fix:要视为需要加载的对象
                zeno::getSession().userData().set2("otherNecessObjs", json_key);
            }
        }
        log_debug("dumping frame {}", frameid);

        if (frameid == beginFrameNumber) {
            std::filesystem::path stampInfoPath = std::filesystem::u8path(cacheFramePath + "/data") / "stampInfo.txt";
            bool hasStampNode = zeno::getSession().userData().has("graphHasStampNode") || std::filesystem::exists(stampInfoPath);
            if (hasStampNode) {
                toDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, runtype, balways, "", true);
            }
        }
        toDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, runtype, balways, "");
    }
}

ZENO_API void GlobalComm::addViewObject(std::string const &key, std::shared_ptr<IObject> object) {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::addViewObject {}", m_frames.size());
    if (m_frames.empty()) throw makeError("empty frame cache");
    m_frames.back().view_objects.try_emplace(key, std::move(object));
}

ZENO_API void GlobalComm::clearState() {
    std::lock_guard lck(m_mtx);
    m_playbackCache.reset();++m_cacheGeneration;
    m_frames.clear();
    m_inCacheFrames.clear();
    m_maxPlayFrame = 0;
    maxCachedFrames = 1;
    cacheFramePath = {};
}

ZENO_API void GlobalComm::clearFrameState()
{
    std::lock_guard lck(m_mtx);
    m_playbackCache.reset();++m_cacheGeneration;
    m_frames.clear();
    m_inCacheFrames.clear();
    m_maxPlayFrame = 0;
}

ZENO_API void GlobalComm::frameCache(std::string const &path, int gcmax) {
    std::lock_guard lck(m_mtx);
    if(path==cacheFramePath && gcmax!=maxCachedFrames){m_playbackCache.reset();++m_cacheGeneration;}
    if(path!=cacheFramePath) {
        m_playbackCache.reset();++m_cacheGeneration;
        m_inCacheFrames.clear();for(auto& frame:m_frames)frame.view_objects.clear();
        assetsInitialized=false;
        for(auto& [scene,flags]:sceneLoadedFlag)std::get<0>(flags)=true;
    }
    cacheFramePath = path;
    maxCachedFrames = gcmax;
}

ZENO_API void GlobalComm::initFrameRange(int beg, int end) {
    std::lock_guard lck(m_mtx);
    if(beg!=beginFrameNumber || end!=endFrameNumber){m_playbackCache.reset();++m_cacheGeneration;}
    beginFrameNumber = beg;
    endFrameNumber = end;
}

ZENO_API int GlobalComm::maxPlayFrames() {
    std::lock_guard lck(m_mtx);
    return m_maxPlayFrame + beginFrameNumber; // m_frames.size();
}

ZENO_API int GlobalComm::numOfFinishedFrame() {
    std::lock_guard lck(m_mtx);
    return m_maxPlayFrame;
}

ZENO_API int GlobalComm::numOfInitializedFrame()
{
    std::lock_guard lck(m_mtx);
    return m_frames.size();
}

ZENO_API std::pair<int, int> GlobalComm::frameRange() {
    std::lock_guard lck(m_mtx);
    return std::pair<int, int>(beginFrameNumber, endFrameNumber);
}

ZENO_API GlobalComm::ViewObjects const *GlobalComm::getViewObjects(const int frameid) {
    std::lock_guard lck(m_mtx);
    bool isLoaded = false, isRerun = false, hasstamp = false;
    std::string runtype;
    return _getViewObjects(frameid, -1, runtype, hasstamp);
}

void GlobalComm::releaseCachedFrame(int frameid) {
    auto cached=m_inCacheFrames.find(frameid);
    if(cached==m_inCacheFrames.end())return;
    const auto index=frameid-beginFrameNumber;
    if(index>=0 && index<m_frames.size()) {
        auto& objects=m_frames[index].view_objects;
        if(m_playbackCache) {
            auto prepared=m_playbackCache->frames.find(frameid);
            if(prepared!=m_playbackCache->frames.end() && prepared->second.resident) {
                objects.m_curr.swap(prepared->second.objects.m_curr);
                cached->second.swap(prepared->second.stamps);
                prepared->second.resident=false;
            }
        }
        objects.clear();
    }
    m_inCacheFrames.erase(cached);
}

GlobalComm::ViewObjects const* GlobalComm::_getViewObjects(const int frameid, uintptr_t sceneIdn, std::string& runtype, bool& hasStamp) {
    bool reloadTimelineFrame=false;
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx < 0 || frameIdx >= m_frames.size())
        return nullptr;
    if (maxCachedFrames != 0) {
        if(m_playbackCache && assetsInitialized && !m_inCacheFrames.empty()) {
            auto prepared=m_playbackCache->frames.find(frameid);
            if(prepared!=m_playbackCache->frames.end()) {
                const auto token=PlaybackCache::token(std::filesystem::u8path(cacheFramePath),frameid);
                if(prepared->second.token==token) {
                    const bool wasResident=m_inCacheFrames.count(frameid)!=0;
                    if(!wasResident) {
                        m_frames[frameIdx].view_objects.m_curr.swap(prepared->second.objects.m_curr);
                        m_inCacheFrames[frameid].swap(prepared->second.stamps);
                        prepared->second.resident=true;
                    }
                    if(!wasResident || currentFrameNumber!=frameid)
                        for(auto& [scene,flags]:sceneLoadedFlag)std::get<2>(flags)=true;
                    runtype=std::get<0>(prepared->second.runInfo);hasStamp=true;currentFrameNumber=frameid;
                    while(m_inCacheFrames.size()>maxCachedFrames) {
                        auto old=m_inCacheFrames.begin();if(old->first==frameid)++old;
                        if(old==m_inCacheFrames.end())break;releaseCachedFrame(old->first);
                    }
                    return &m_frames[frameIdx].view_objects;
                }
                const bool assetsChanged=!token.valid || token.files.back()!=prepared->second.token.files.back();
                while(!m_inCacheFrames.empty())releaseCachedFrame(m_inCacheFrames.begin()->first);
                m_playbackCache.reset();++m_cacheGeneration;
                if(assetsChanged) {
                    assetsInitialized=false;
                    for(auto& [scene,flags]:sceneLoadedFlag)std::get<0>(flags)=true;
                } else reloadTimelineFrame=true;
            }
        }
        // load back one gc:
        if (!m_inCacheFrames.count(frameid)) {  // notinmem then cacheit
            std::get<2>(sceneLoadedFlag[sceneIdn]) = true;

            std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>> baseframeinfo;

            auto runinfo = getRunType(std::filesystem::u8path(cacheFramePath) / std::to_string(1000000 + frameid).substr(1));
            runtype = std::get<0>(runinfo);
            bool always = std::get<1>(runinfo);
            std::string currViewNodes = std::get<2>(runinfo);
            std::filesystem::path stampInfoPath = std::filesystem::u8path(cacheFramePath + "/" + std::to_string(1000000 + frameid).substr(1)) / "stampInfo.txt";
            if (std::filesystem::exists(stampInfoPath)) {
                if (m_inCacheFrames.empty()) {//重新运行了
                    if (!assetsInitialized || runtype == "LoadAsset") {//
                        bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, false, true);
                        if (!ret)
                            return nullptr;
                        //for (auto& [k, obj] : m_frames[frameIdx].view_objects.m_curr) {
                        //    obj->userData().set2("stamp-change", "TotalChange");
                        //}
                        if (!assetsInitialized) {
                            assetsInitialized = true;
                        } else if (runtype == "LoadAsset") {
                            for (auto& [objPtrId, flag] : sceneLoadedFlag) {
                                std::get<0>(flag) = true;
                            }
                        }
                    } else {//除assetload外的运行,按实际stamp加载
                        for (auto& [objPtrId, flag] : sceneLoadedFlag) {
                            std::get<1>(flag) = true;
                        }
                        //if (currentFrameNumber != frameid) {//运行时不在起始帧,发生了切帧
                        //    runtype = "RunAll";
                        //}
                        bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, reloadTimelineFrame, false, currViewNodes);
                        if (!ret)
                            return nullptr;
                    }
                } else {//切帧
                    for (auto& [objPtrId, flag] : sceneLoadedFlag) {
                        std::get<2>(flag) = true;
                    }
                    //runtype = runtype != "LoadAsset" && runtype != "RunAll" ? "RunAll" : runtype; //切帧要加载新帧的全部
                    bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, true, false, currViewNodes);
                    if (!ret)
                        return nullptr;
                }
                hasStamp = true;
            } else {
                bool ret = fromDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, runtype);
                if (!ret)
                    return nullptr;
                hasStamp = false;
            }

            m_inCacheFrames.insert({frameid, baseframeinfo });
            // and dump one as balance:
            if (m_inCacheFrames.size() && m_inCacheFrames.size() > maxCachedFrames) { // notindisk then dumpit
                for (auto& [i, _] : m_inCacheFrames) {
                    if (i != frameid) {
                        // seems that objs will not be modified when load_objects called later.
                        // so, there is no need to dump.
                        //toDisk(cacheFramePath, i, m_frames[i - beginFrameNumber].view_objects);
                        releaseCachedFrame(i);
                        break;
                    }
                }
            }
        } else {
             if (currentFrameNumber != frameid) {
                std::filesystem::path stampInfoPath = std::filesystem::u8path(cacheFramePath + "/" + std::to_string(1000000 + frameid).substr(1)) / "stampInfo.txt";
                if (std::filesystem::exists(stampInfoPath)) {
                    for (auto& [objPtrId, flag] : sceneLoadedFlag) {
                        std::get<2>(flag) = true;
                    }
                    if (frameid == beginFrameNumber) {
                        std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>> baseframeinfo;
                        m_frames[frameIdx].view_objects.m_curr.clear();

                        auto runinfo = getRunType(std::filesystem::u8path(cacheFramePath) / std::to_string(1000000 + frameid).substr(1));
                        runtype = std::get<0>(runinfo);
                        std::string currViewNodes = std::get<2>(runinfo);
                        bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, true, false, currViewNodes);
                        if (!ret)
                            return nullptr;
                        m_inCacheFrames[frameid] = baseframeinfo;
                    }
                    hasStamp = true;
                }
                else {
                    hasStamp = false;
                }
            }
        }
    }
    currentFrameNumber = frameid;
    return &m_frames[frameIdx].view_objects;
}

ZENO_API GlobalComm::ViewObjects const &GlobalComm::getViewObjects() {
    std::lock_guard lck(m_mtx);
    return m_frames.back().view_objects;
}

ZENO_API void GlobalComm::clear_objects(const std::function<void()>& callback)
{
    std::lock_guard lck(m_mtx);
    if (!callback)
        return;

    callback();
}


ZENO_API bool GlobalComm::load_objects(
        const int frameid,
        const std::function<bool(std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs, std::string& runtype)>& callback,
        std::function<void(int frameid, bool inserted, bool hasstamp)> callbackUpdate,
        uintptr_t sceneId,
        bool& isFrameValid)
{
    if (!callback)
        return false;

    std::lock_guard lck(m_mtx);

    int frame = frameid;
    frame -= beginFrameNumber;
    if (frame < 0 || frame >= m_frames.size() || m_frames[frame].frame_state != FRAME_COMPLETED)
    {
        isFrameValid = false;
        return false;
    }

    isFrameValid = true;
    bool inserted = false;
    static bool hasstamp = false;
    static std::string runtype;
    auto const* viewObjs = _getViewObjects(frameid, sceneId, runtype, hasstamp);
    if (viewObjs) {
        zeno::log_trace("load_objects: {} objects at frame {}", viewObjs->size(), frameid);
        inserted = callback(viewObjs->m_curr, runtype);
    }
    else {
        zeno::log_trace("load_objects: no objects at frame {}", frameid);
        inserted = callback({}, runtype);
    }
    callbackUpdate(frameid, inserted, hasstamp);
    return inserted;
}

ZENO_API bool GlobalComm::isFrameCompleted(int frameid) const {
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return false;
    return m_frames[frameid].frame_state == FRAME_COMPLETED;
}

ZENO_API GlobalComm::FRAME_STATE GlobalComm::getFrameState(int frameid) const
{
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return FRAME_UNFINISH;
    return m_frames[frameid].frame_state;
}

ZENO_API bool GlobalComm::isFrameBroken(int frameid) const
{
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return false;
    return m_frames[frameid].frame_state == FRAME_BROKEN;
}

ZENO_API int GlobalComm::maxCachedFramesNum()
{
    std::lock_guard lck(m_mtx);
    return maxCachedFrames;
}

ZENO_API std::string GlobalComm::cachePath()
{
    std::lock_guard lck(m_mtx);
    return cacheFramePath;
}

ZENO_API bool GlobalComm::removeCache(int frame)
{
    std::lock_guard lck(m_mtx);
    bool hasZencacheOnly = true;
    std::filesystem::path dirToRemove = std::filesystem::u8path(cacheFramePath + "/" + std::to_string(1000000 + frame).substr(1));
    if (std::filesystem::exists(dirToRemove))
    {
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dirToRemove))
        {
            std::string filePath = entry.path().string();
            std::string fileName = entry.path().filename().string();
            if (std::filesystem::is_directory(entry.path()) || (filePath.substr(filePath.size() - 9) != ".zencache" && fileName != "runInfo.txt" && fileName != "stampInfo.txt"))
            {
                hasZencacheOnly = false;
                break;
            }
        }
        if (hasZencacheOnly)
        {
            m_frames[frame - beginFrameNumber].frame_state = FRAME_BROKEN;
            std::filesystem::remove_all(dirToRemove);
            zeno::log_info("remove dir: {}", dirToRemove);
        }
    }
    if (frame == endFrameNumber && std::filesystem::exists(std::filesystem::u8path(cacheFramePath)))
    {
        std::filesystem::path dataDir = std::filesystem::u8path(cacheFramePath + "/data");
        if (std::filesystem::exists(dataDir)) {
            std::filesystem::remove_all(dataDir);
            zeno::log_info("remove dir: {}", dataDir);
        }
        if (std::filesystem::is_empty(std::filesystem::u8path(cacheFramePath))) {
            std::filesystem::remove(std::filesystem::u8path(cacheFramePath));
            zeno::log_info("remove dir: {}", std::filesystem::u8path(cacheFramePath).string());
        }
    }
    return true;
}

ZENO_API void GlobalComm::removeCachePath()
{
    std::lock_guard lck(m_mtx);
    std::filesystem::path dirToRemove = std::filesystem::u8path(cacheFramePath);
    if (std::filesystem::exists(dirToRemove) && cacheFramePath.find(".") == std::string::npos)
    {
        std::filesystem::remove_all(dirToRemove);
        zeno::log_info("remove dir: {}", dirToRemove);
    }
}

ZENO_API std::string GlobalComm::cacheTimeStamp(int frame, bool& exists)
{
    std::lock_guard lck(m_mtx);
    int framesize = frame - beginFrameNumber;
    if (m_frames.size() <= framesize || m_frames[framesize].frame_state != FRAME_COMPLETED) {
        return "";
    }
    auto dir = std::filesystem::u8path(cacheFramePath) / std::to_string(1000000 + frame).substr(1);
    if (std::filesystem::exists(dir)) {
        if (!std::filesystem::is_empty(dir)) {
            auto lockfilename = zeno::iotags::sZencache_lockfile_prefix + std::to_string(frame) + ".lock";
            if (!std::filesystem::exists(std::filesystem::u8path(cacheFramePath) / lockfilename)) {
                std::ostringstream oss;
                for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
                    if (std::filesystem::is_regular_file(entry)) {
                        try {
                            const std::filesystem::path& filePath = entry.path();
                            auto ftime = std::filesystem::last_write_time(filePath);

                            auto sctp = std::chrono::time_point_cast<std::chrono::milliseconds>(ftime);
                            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(sctp.time_since_epoch()).count();
                            oss << timestamp;
                        }
                        catch (const std::filesystem::filesystem_error& e) {
                            return "";
                        }
                    }
                }
                std::string res = oss.str();
                if (res.empty()) {
                    return "";
                } else {
                    exists = true;
                    return oss.str();
                }
            }
        }
    }
    return "";
}

ZENO_API std::string GlobalComm::getBenchmarkLog()
{
    return Timer::getLog();
}

}
