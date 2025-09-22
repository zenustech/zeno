#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <filesystem>
#include <algorithm>
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

    std::filesystem::path dir = std::filesystem::u8path(cachedir + "/" + (isStampModeInit ? "data" : std::to_string(1000000 + frameid).substr(1)));
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir))
    {
        log_critical("can not create path: {}", dir);
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
                        if (group.name.GetString() == runtype) {
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
            int baseframe = isStampModeInit ? -9999 : (stamptag == "TotalChange" || frameid == beginFrameNumber ? frameid : std::get<1>(lastframeStampinfo[key.substr(0, secondLastColonIdx(key))]));
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
					unchangeObj->m_userData = std::move(obj->userData());
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
    bool loadPartial = false;

    auto dir = std::filesystem::u8path(cachedir) / (loadasset ? "data" : std::to_string(1000000 + frameid).substr(1));
    //runtype = getRunType(dir);

    std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>> currentFrameStampinfo;
    auto it = m_inCacheFrames.find(currentFrameNumber);
    if (it != m_inCacheFrames.end()) {
        currentFrameStampinfo = it->second;
    }

    std::filesystem::path frameStampPath = dir / "stampInfo.txt";
    std::ifstream file(frameStampPath);
    if (file) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        rapidjson::Document doc, datadoc;
        doc.Parse(buffer.str().c_str());
        if (doc.IsObject()) {
            std::vector<rapidjson::Value> list(4);
            if (!switchTimeline && !loadasset) {//不是loadasset的运行分开加载stampinfo
                auto datadir = std::filesystem::u8path(cachedir) / "data";
                std::ifstream datafile(datadir / "stampInfo.txt");
                if (datafile) {
                    std::stringstream databuffer;
                    databuffer << datafile.rdbuf();
                    datadoc.Parse(databuffer.str().c_str());
                    if (datadoc.IsObject()) {
                        if (runtype == "RunAll") {
                            list[0] = datadoc.GetObject()["lightCameraObj"].GetObject();
                            list[1] = datadoc.GetObject()["materialObj"].GetObject();
                            list[2] = datadoc.GetObject()["matrixObj"].GetObject();
                        } else if (runtype == "RunLightCamera") {
							list[0] = datadoc.GetObject()["lightCameraObj"].GetObject();
							list[1] = doc.GetObject()["materialObj"].GetObject();
							list[2] = doc.GetObject()["matrixObj"].GetObject();
                        } else if (runtype == "RunMaterial") {
							list[0] = doc.GetObject()["lightCameraObj"].GetObject();
							list[1] = datadoc.GetObject()["materialObj"].GetObject();
							list[2] = doc.GetObject()["matrixObj"].GetObject();
                        } else if (runtype == "RunMatrix") {
							list[0] = doc.GetObject()["lightCameraObj"].GetObject();
							list[1] = doc.GetObject()["materialObj"].GetObject();
							list[2] = datadoc.GetObject()["matrixObj"].GetObject();
                        }
					    list[3] = doc.GetObject()["normalObj"].GetObject();
                    }
                }
            } else {
                list[0] = doc.GetObject()["lightCameraObj"].GetObject();
                list[1] = doc.GetObject()["materialObj"].GetObject();
                list[2] = doc.GetObject()["matrixObj"].GetObject();
                list[3] = doc.GetObject()["normalObj"].GetObject();
            }
            if (runtype == "RunAll" || runtype == "LoadAsset") {
                for (auto& val : list) {
                    for (const auto& node : val.GetObject()) {
                        const std::string& newFrameChangeInfo = node.value.HasMember("stamp-change") ? node.value["stamp-change"].GetString() : "TotalChange" ;
                        const int& newFrameBaseframe = node.value.HasMember("stamp-base") ? node.value["stamp-base"].GetInt() : -1;
                        const int& newFrameObjtype = node.value.HasMember("stamp-objType") ? node.value["stamp-objType"].GetInt() : 0;
                        const std::string& newFrameObjkey = node.name.GetString();
                        const size_t& newFrameObjStartIdx = node.value.HasMember("startIndexInCache") ? std::stoull(node.value["startIndexInCache"].GetString()) : 0;
                        const size_t& newFrameObjLength = node.value.HasMember("ObjSize") ? std::stoull(node.value["ObjSize"].GetString()) : 0;

                        //const std::string& nodeid = newFrameObjkey.substr(0, newFrameObjkey.find_first_of(":"));
                        const std::string& nodeid = newFrameObjkey.substr(0, secondLastColonIdx(newFrameObjkey));
                        const std::string& newFrameChangeHint = node.value.HasMember("stamp-dataChange-hint") ? node.value["stamp-dataChange-hint"].GetString() : "";
                        newFrameStampInfo.insert({ nodeid , std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>({newFrameChangeInfo, newFrameBaseframe, newFrameObjtype, newFrameObjkey, newFrameChangeHint, newFrameObjStartIdx, newFrameObjLength})});
                        if (!currentFrameStampinfo.empty()) {
                            const std::string& curFrameChangeInfo = std::get<0>(currentFrameStampinfo[nodeid]);
                            const int& curFrameBaseframe = std::get<1>(currentFrameStampinfo[nodeid]);

                            if (curFrameBaseframe != newFrameBaseframe) {
                                if (newFrameChangeInfo != "TotalChange") {//不是Totalchange但baseframe变化的情况
                                    loadPartial = true;
                                }
                            }
                        }
        #if 0
                        else {//currentFrameStampinfo为空是重新run
                            if (newFrameChangeInfo != "TotalChange") {//重新run且不是Totalchange
                                loadPartial = true;
                            }
                        }
        #endif
                    }
                }
            } else { 
                if (runtype == "RunLightCamera") {
                } else if (runtype == "RunMaterial") {
                    std::swap(list[0], list[1]);
                } else if (runtype == "RunMatrix") {
                    std::swap(list[0], list[2]);
                }
                std::string objtype = runtype == "RunLightCamera" ? "lightCameraObj" : (runtype == "RunMaterial" ? "materialObj" : (runtype == "RunMatrix" ? "matrixObj" : "normalObj"));
                for (int i = 0; i < 4; i++) {
                    for (const auto& node : list[i].GetObject()) {
                        const std::string& newFrameObjkey = node.name.GetString();
                        const std::string& nodeid = newFrameObjkey.substr(0, secondLastColonIdx(newFrameObjkey));
                        int objRunType = (node.value.HasMember("objRunType") ? node.value["objRunType"].GetInt() : 3);
                        std::string correspondruntype = objRunType == 0 ? "RunLightCamera" : (objRunType == 1 ? "RunMaterial" : (objRunType == 2 ? "RunMatrix" : "RunAll"));

                        if (newFrameStampInfo.find(nodeid) == newFrameStampInfo.end() &&
                            currViewNodes.find(nodeid.substr(0, nodeid.find_first_of(':'))) != std::string::npos) {//当运行灯光相机材质矩阵时，必须存在在当前的viewNodes中才能加载（避免加载到旧的obj）
                            const std::string& newFrameChangeInfo = node.value.HasMember("stamp-change") ? node.value["stamp-change"].GetString() : "TotalChange";
                            const int& newFrameBaseframe = node.value.HasMember("stamp-base") ? node.value["stamp-base"].GetInt() : -1;
                            const int& newFrameObjtype = node.value.HasMember("stamp-objType") ? node.value["stamp-objType"].GetInt() : 0;
                            const size_t& newFrameObjStartIdx = node.value.HasMember("startIndexInCache") ? std::stoull(node.value["startIndexInCache"].GetString()) : 0;
                            const size_t& newFrameObjLength = node.value.HasMember("ObjSize") ? std::stoull(node.value["ObjSize"].GetString()) : 0;

                            const std::string& newFrameChangeHint = node.value.HasMember("stamp-dataChange-hint") ? node.value["stamp-dataChange-hint"].GetString() : "";
                            newFrameStampInfo.insert({ nodeid , std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>({newFrameChangeInfo, newFrameBaseframe, newFrameObjtype, newFrameObjkey, newFrameChangeHint, newFrameObjStartIdx, newFrameObjLength}) });

                        }
                    }
                }
            }
        }
    }
    //除assetload之外的运行(涉及lightcamera,material,matrix)，如果是UnChanged，需要标记为totalchange以便光追能更新(因为可能改了数据)
    const auto& needMarkAsTotalChange = [&switchTimeline, &runtype](int cachetype, std::string stampchange) -> bool {
        if (!switchTimeline && stampchange == "UnChanged") {
            if (
                //runtype == "RunAll" && (cachetype == 0 || cachetype == 1 || cachetype == 2) ||
                runtype == "RunLightCamera" && cachetype == 0 ||
                runtype == "RunMaterial" && cachetype == 1 || 
                runtype == "RunMatrix" && cachetype == 2) {
                return true;
            }
        }
        return false;
    };
    const auto& load = [&dir,&needMarkAsTotalChange, &newFrameStampInfo, &switchTimeline, &runtype, &loadasset](std::string cachedir, GlobalComm::ViewObjects& objs, std::string& runtype)->bool {
        if (cachedir.empty())
            return false;

        if (!switchTimeline && !loadasset) {//不是loadasset的运行分开加载stampinfo
			auto datadir = std::filesystem::u8path(cachedir) / "data";
			if (runtype == "RunAll") {
                cachepath[0] = datadir / "lightCameraObj.zencache";
                cachepath[1] = datadir / "materialObj.zencache";
                cachepath[2] = datadir / "matrixObj.zencache";
			}
			else if (runtype == "RunLightCamera") {
                cachepath[0] = datadir / "lightCameraObj.zencache";
                cachepath[1] = dir / "materialObj.zencache";
                cachepath[2] = dir / "matrixObj.zencache";
			}
            else if (runtype == "RunMaterial") {
                cachepath[0] = datadir / "materialObj.zencache";
                cachepath[1] = dir / "lightCameraObj.zencache";
                cachepath[2] = dir / "matrixObj.zencache";
            }
            else if (runtype == "RunMatrix") {
                cachepath[0] = datadir / "matrixObj.zencache";
                cachepath[1] = dir / "lightCameraObj.zencache";
                cachepath[2] = dir / "materialObj.zencache";
            }
			cachepath[3] = dir / "normalObj.zencache";
        } else {
            cachepath[0] = dir / "lightCameraObj.zencache";
            cachepath[1] = dir / "materialObj.zencache";
            cachepath[2] = dir / "matrixObj.zencache";
            cachepath[3] = dir / "normalObj.zencache";
        }

        for (int i = 0; i < cachepath.size(); i++)
        {
            //if (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2) {
            //    continue;
            //}
            if (!std::filesystem::exists(cachepath[i])) {
                continue;
            }
            log_debug("load cache from disk {}", cachepath[i]);

            auto szBuffer = std::filesystem::file_size(cachepath[i]);

            std::ifstream file(cachepath[i], std::ios::binary);
            if (!file.is_open()) {
                log_error("zeno cache file does not exist");
                return false;
            }
            std::string keysStr;
            std::getline(file, keysStr, '\a');
            size_t pos = keysStr.size();
            if (pos <= 8) {
                log_error("zeno cache file broken");
                return false;
            }

            size_t keyscount = std::stoi(keysStr.substr(8));
            if (keyscount < 1) {
                continue;
            }

            std::vector<std::string>keys(keyscount);
            for (int k = 0; k < keyscount; k++) {
                std::getline(file, keys[k], '\a');
            }
            file.seekg((keyscount + 1) * sizeof(size_t), std::ios::cur);
            for (auto& k : keys) {
                //auto it = newFrameStampInfo.find(k.substr(0, k.find_first_of(":")));
                auto it = newFrameStampInfo.find(k.substr(0, secondLastColonIdx(k)));
                if (it != newFrameStampInfo.end()) {
                    bool needMarkTotalChange = needMarkAsTotalChange(i, std::get<0>(it->second));
                    if (std::get<0>(it->second) == "TotalChange" || needMarkTotalChange) {
                        int64_t originalPos = file.tellg();
                        file.seekg(std::get<5>(it->second), std::ios::cur);
                        int64_t startPos = file.tellg();
                        if(startPos + std::get<6>(it->second) > szBuffer) {
                            continue;
                        }
                        std::vector<char> objbuff(std::get<6>(it->second));
                        file.read(objbuff.data(), std::get<6>(it->second));
                        file.seekg(originalPos);
                        auto spobj = decodeObject(objbuff.data(), std::get<6>(it->second));
                        if (spobj && needMarkTotalChange) {
                            spobj->userData().set2("stamp-change", "TotalChange");
                        }
                        objs.try_emplace(std::get<3>(it->second), spobj);
                    } else if (std::get<0>(it->second) == "UnChanged") {
						int64_t originalPos = file.tellg();
						file.seekg(std::get<5>(it->second), std::ios::cur);
						int64_t startPos = file.tellg();
						if (startPos + std::get<6>(it->second) > szBuffer) {
							continue;
						}
						std::vector<char> objbuff(std::get<6>(it->second));
						file.read(objbuff.data(), std::get<6>(it->second));
						file.seekg(originalPos);
						auto spobj = decodeObject(objbuff.data(), std::get<6>(it->second));
						if (spobj) {
							spobj->userData().set2("stamp-change", "UnChanged");
						}
						objs.try_emplace(std::get<3>(it->second), spobj);
					}
                }
            }
        }
        return true;
    };
    //if (!loadPartial) {
        bool ret = load(cacheFramePath, objs, runtype);
//        for (auto& [key, tup] : newFrameStampInfo) {
//            //if (std::get<0>(tup) == "UnChanged") {
//            if (std::get<0>(tup) != "TotalChange") {//不是Totalchange的，暂时全部按照unchange处理
//                std::shared_ptr<IObject> emptyobj = constructEmptyObj(std::get<2>(tup));
//                emptyobj->userData().set2("stamp-change", std::get<0>(tup));
//                emptyobj->userData().set2("stamp-base", std::get<1>(tup));
//                objs.try_emplace(std::get<3>(tup), emptyobj);
//            }
//        }
        return ret;
    //}
#if 0
    else {
        bool ret = load(cacheFramePath, objs, runtype);
        if (ret) {
            for (auto& [key, tup] : newFrameStampInfo) {
                int newframeObjBaseframe = std::get<1>(tup);
                std::string newframeObjfullkey = std::get<3>(tup);
                std::string newframeObjStampchange = std::get<0>(tup);
                std::string newframeDataChangeHint = std::get<4>(tup);

                if (newframeObjStampchange == "UnChanged") {
                    if (auto correspondFrameObj = fromDiskReadObject(cacheFramePath, newframeObjBaseframe, key)) {
                        correspondFrameObj->userData().set2("stamp-change", "TotalChange");
                        objs.try_emplace(key + newframeObjfullkey.substr(lasttwo(newframeObjfullkey)), correspondFrameObj);
                    }
                }
                else if (newframeObjStampchange == "DataChange") {
                    auto baseobj = std::move(fromDiskReadObject(cacheFramePath, newframeObjBaseframe, key));
                    auto newDataChangedObj = objs.m_curr[newframeObjfullkey];
                    //根据newframeDataChangeHint获取newDataChangedObj的data,设置给baseobj
                    objs.m_curr.erase(newframeObjfullkey);
                    objs.try_emplace(key + newframeObjfullkey.substr(lasttwo(newframeObjfullkey)), baseobj);
                }
                else if (newframeObjStampchange == "ShapeChange") {
                    auto baseobj = std::move(fromDiskReadObject(cacheFramePath, newframeObjBaseframe, key));
                    auto newDataChangedObj = objs.m_curr[newframeObjfullkey];
                    //暂时并入Totalchange
                    objs.m_curr.erase(newframeObjfullkey);
                    objs.try_emplace(key + newframeObjfullkey.substr(lasttwo(newframeObjfullkey)), baseobj);
                }
            }
            return true;
        }
        return ret;//此时objs中对象的stamp-change已经根据切帧前后的变化正确调整
    }
#endif
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
            std::get<2>(runinfo) = obj["viewNodes"].GetString();
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
            }
            if (!dynamic_scene_descriptor.is_null()) {
                auto &json = dynamic_scene_descriptor;
                if (!json["BasicRenderInstances"].is_null()) {
                    scene_descriptor_json["BasicRenderInstances"].update(json["BasicRenderInstances"]);
                }
                if (json.contains("DynamicRenderGroups") && !json["DynamicRenderGroups"].is_null()) {
                    scene_descriptor_json["DynamicRenderGroups"].update(json["DynamicRenderGroups"]);
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
    m_frames.clear();
    m_inCacheFrames.clear();
    m_maxPlayFrame = 0;
    maxCachedFrames = 1;
    cacheFramePath = {};
}

ZENO_API void GlobalComm::clearFrameState()
{
    std::lock_guard lck(m_mtx);
    m_frames.clear();
    m_inCacheFrames.clear();
    m_maxPlayFrame = 0;
}

ZENO_API void GlobalComm::frameCache(std::string const &path, int gcmax) {
    std::lock_guard lck(m_mtx);
    cacheFramePath = path;
    maxCachedFrames = gcmax;
}

ZENO_API void GlobalComm::initFrameRange(int beg, int end) {
    std::lock_guard lck(m_mtx);
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

GlobalComm::ViewObjects const* GlobalComm::_getViewObjects(const int frameid, uintptr_t sceneIdn, std::string& runtype, bool& hasStamp) {
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx < 0 || frameIdx >= m_frames.size())
        return nullptr;
    if (maxCachedFrames != 0) {
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
                        bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, false, false, currViewNodes);
                        if (!ret)
                            return nullptr;
                    }
                } else {//切帧
                    for (auto& [objPtrId, flag] : sceneLoadedFlag) {
                        std::get<2>(flag) = true;
                    }
                    //runtype = runtype != "LoadAsset" && runtype != "RunAll" ? "RunAll" : runtype; //切帧要加载新帧的全部
                    bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, true);
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
                        m_frames[i - beginFrameNumber].view_objects.clear();
                        m_inCacheFrames.erase(i);
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
                        bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype, true);
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
