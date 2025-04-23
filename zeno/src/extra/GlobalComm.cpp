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

std::vector<std::filesystem::path> cachepath(4);
void GlobalComm::toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs, std::string runtype, std::string fileName, bool isBeginframe) {
    if (cachedir.empty()) return;

    const auto& lasttwo = [](const std::string& str)->int {
        int count = 0;
        for (int i = str.length() - 1; i >= 0; --i) {
            if (str[i] == ':') {
                ++count;
                if (count == 2)
                    return i;
            }
        }
        return -1;
    };

    std::filesystem::path dir = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid).substr(1));
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir))
    {
        log_critical("can not create path: {}", dir);
    }

    std::filesystem::path runInfoPath = dir / "runInfo.txt";
    std::ofstream runinfoofs(runInfoPath, std::ios::binary);
    std::ostreambuf_iterator<char> runinfooit(runinfoofs);
    std::copy(runtype.begin(), runtype.end(), runinfooit);

    std::vector<std::vector<char>> bufCaches(4);
    std::vector<std::vector<size_t>> poses(4);
    std::vector<std::string> keys(4);

    std::filesystem::path stampInfoPath = dir / "stampInfo.txt";
    bool hasStampNode = zeno::getSession().userData().has("graphHasStampNode") || std::filesystem::exists(stampInfoPath);
    if (hasStampNode) {
        std::filesystem::path stampInfoPath = dir / "stampInfo.txt";
        std::map<std::string, std::tuple<std::string, int>> lastframeStampinfo;
        std::string docKey = runtype == "LoadAsset" ? "RunAll" : runtype;

        if (!isBeginframe) {
            rapidjson::Document doc;
            std::filesystem::path lastframeStampPath = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid - 1).substr(1)) / "stampInfo.txt";
            std::ifstream file(lastframeStampPath);
            if (file) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                doc.Parse(buffer.str().c_str());
                if (doc.IsObject()) {
                    for (const auto& group : doc.GetObject()) {
                        if (group.name.GetString() == docKey) {
                            for (const auto& node : group.value.GetObject()) {
                                const std::string& key = node.name.GetString();
                                lastframeStampinfo.insert({ key.substr(0, lasttwo(key)), std::tuple<std::string, int>(node.value["stamp-change"].GetString(), node.value["stamp-base"].GetInt()) });
                            }
                        }
                    }
                }
            }
        }
        rapidjson::Document currFrameStampInfodoc;
        std::filesystem::path lastframeStampPath = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid).substr(1)) / "stampInfo.txt";
        std::ifstream currFrameiffile(lastframeStampPath);
        std::stringstream currFramestampinfobuffer;
        currFramestampinfobuffer << currFrameiffile.rdbuf();
        currFrameStampInfodoc.Parse(currFramestampinfobuffer.str().c_str());

        rapidjson::StringBuffer str;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(str);
        writer.StartObject();
        if (currFrameStampInfodoc.IsObject()) {
            for (auto& member : currFrameStampInfodoc.GetObject()) {
                const char* key = member.name.GetString();
                if (std::string(key) != docKey) {
                    const rapidjson::Value& value = member.value;
                    writer.Key(key);
                    value.Accept(writer);
                }
            }
        }
        writer.Key(docKey.c_str());
        writer.StartObject();
        for (auto& [key, obj] : objs) {
            /*if (isBeginframe) {
                obj->userData().set2("stamp-change", "TotalChange");
            }*/
            const std::string& stamptag = obj->userData().get2<std::string>("stamp-change", "TotalChange");
            const int& baseframe = stamptag == "TotalChange" || isBeginframe ? frameid : std::get<1>(lastframeStampinfo[key.substr(0, lasttwo(key))]);
            obj->userData().set2("stamp-base", baseframe);
            obj->userData().set2("stamp-change", stamptag);


            //写出stampinfo
            writer.Key(key.c_str());
            writer.StartObject();
            writer.Key("stamp-change");
            writer.String(stamptag.c_str());
            writer.Key("stamp-base");
            writer.Int(baseframe);

            if (0) {
    #define _PER_OBJECT_TYPE(TypeName, ...) \
            } else if (auto o = dynamic_cast<TypeName const *>(obj.get())) { \
                writer.Key("stamp-objType"); \
                writer.Int((int)ObjectType::TypeName);
            ZENO_XMACRO_IObject(_PER_OBJECT_TYPE)
    #undef _PER_OBJECT_TYPE
            } else {
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
                if (!isBeginframe) {
                    writer.EndObject();
                    continue;//不是起始帧的unchange，不输出这个obj
                }
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

            //写出stampinfo的obj尺寸
            writer.Key("startIndexInCache");
            writer.String(std::to_string(bufsize).c_str());
            writer.Key("ObjSize");
            writer.String(std::to_string(bufCaches[idx].size() - bufsize).c_str());
            writer.EndObject();
        }
        //写出stampinfo
        writer.EndObject();
        writer.EndObject();
        std::string stampinfos = str.GetString();
        std::ofstream ofs(stampInfoPath, std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(stampinfos.begin(), stampinfos.end(), oit);
    } else {
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
        if (poses[i].size() == 0 && (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2))
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
        if (poses[i].size() == 0 && (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2))
            continue;
        log_debug("dump cache to disk {}", cachepath[i]);
        std::ofstream ofs(cachepath[i], std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(keys[i].begin(), keys[i].end(), oit);
        std::copy_n((const char*)poses[i].data(), poses[i].size() * sizeof(size_t), oit);
        std::copy(bufCaches[i].begin(), bufCaches[i].end(), oit);
    }

    objs.clear();
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

    for (int i = 0; i < cachepath.size(); i++)
    {
        if (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2) {
            continue;
        }
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
            objs.try_emplace(keys[k], decodeObject(p, poses[k + 1] - poses[k]));
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

bool GlobalComm::fromDiskByStampinfo(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>>& newFrameStampInfo, std::string& runtype)
{
    const auto& lasttwo = [](const std::string& str)->int {
        int count = 0;
        for (int i = str.length() - 1; i >= 0; --i) {
            if (str[i] == ':') {
                ++count;
                if (count == 2)
                    return i;
            }
        }
        return -1;
    };
    bool loadPartial = false;

    auto dir = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1);
    //runtype = getRunType(dir);

    std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>> currentFrameStampinfo;
    auto it = m_inCacheFrames.find(currentFrameNumber);
    if (it != m_inCacheFrames.end()) {
        currentFrameStampinfo = it->second;
    }

    std::filesystem::path frameStampPath = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid).substr(1)) / "stampInfo.txt";
    std::ifstream file(frameStampPath);
    if (file) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        rapidjson::Document doc;
        doc.Parse(buffer.str().c_str());
        if (doc.IsObject()) {
            for (const auto& group : doc.GetObject()) {
                if (group.name.GetString() == runtype) {
                    for (const auto& node : group.value.GetObject()) {
                        const std::string& newFrameChangeInfo = node.value["stamp-change"].GetString();
                        const int& newFrameBaseframe = node.value["stamp-base"].GetInt();
                        const int& newFrameObjtype = node.value["stamp-objType"].GetInt();
                        const std::string& newFrameObjkey = node.name.GetString();
                        const size_t& newFrameObjStartIdx = node.value.HasMember("startIndexInCache") ? std::stoull(node.value["startIndexInCache"].GetString()) : 0;
                        const size_t& newFrameObjLength = node.value.HasMember("ObjSize") ? std::stoull(node.value["ObjSize"].GetString()) : 0;
                
                        //const std::string& nodeid = newFrameObjkey.substr(0, newFrameObjkey.find_first_of(":"));
                        const std::string& nodeid = newFrameObjkey.substr(0, lasttwo(newFrameObjkey));
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
            }
        }
    }
    const auto& load = [&lasttwo, &dir, &newFrameStampInfo](std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::string& runtype)->bool {
        if (cachedir.empty())
            return nullptr;

        cachepath[0] = dir / "lightCameraObj.zencache";
        cachepath[1] = dir / "materialObj.zencache";
        cachepath[2] = dir / "matrixObj.zencache";
        cachepath[3] = dir / "normalObj.zencache";

        for (int i = 0; i < cachepath.size(); i++)
        {
            if (runtype == "RunLightCamera" && i != 0 || runtype == "RunMaterial" && i != 1 || runtype == "RunMatrix" && i != 2) {
                continue;
            }
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
                auto it = newFrameStampInfo.find(k.substr(0, lasttwo(k)));
                if (it != newFrameStampInfo.end()) {
                    if (std::get<0>(it->second) == "TotalChange") {
                        std::streampos originalPos = file.tellg();
                        file.seekg(std::get<5>(it->second), std::ios::cur);
                        std::vector<char> objbuff(std::get<6>(it->second));
                        file.read(objbuff.data(), std::get<6>(it->second));
                        file.seekg(originalPos);
                        objs.try_emplace(std::get<3>(it->second), decodeObject(objbuff.data(), std::get<6>(it->second)));
                    }
                }
            }
        }
        return true;
    };
    //if (!loadPartial) {
        bool ret = load(cacheFramePath, frameid, objs, runtype);
        for (auto& [key, tup] : newFrameStampInfo) {
            if (std::get<0>(tup) == "UnChanged") {
                std::shared_ptr<IObject> emptyobj = constructEmptyObj(std::get<2>(tup));
                emptyobj->userData().set2("stamp-change", std::get<0>(tup));
                emptyobj->userData().set2("stamp-base", std::get<1>(tup));
                objs.try_emplace(std::get<3>(tup), emptyobj);
            }
        }
        return ret;
    //}
#if 0
    else {
        bool ret = load(cacheFramePath, frameid, objs, runtype);
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

    std::string runtype = getRunType(dir);

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

std::string GlobalComm::getRunType(std::filesystem::path dir)
{
    std::filesystem::path runinfoPath = dir / "runInfo.txt";
    std::ifstream runinfoFile(runinfoPath, std::ios::binary);
    if (!runinfoFile.is_open()) {
        log_error("run info file does not exist");
        return "RunAll";
    }
    std::string runtype;
    std::getline(runinfoFile, runtype);
    return runtype;
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

ZENO_API void GlobalComm::dumpFrameCache(int frameid, std::string runtype) {
    std::lock_guard lck(m_mtx);
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx >= 0 && frameIdx < m_frames.size()) {
        log_debug("dumping frame {}", frameid);
        toDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, runtype, "", frameid == beginFrameNumber);
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

            runtype = getRunType(std::filesystem::u8path(cacheFramePath) / std::to_string(1000000 + frameid).substr(1));
            std::filesystem::path stampInfoPath = std::filesystem::u8path(cacheFramePath + "/" + std::to_string(1000000 + frameid).substr(1)) / "stampInfo.txt";
            if (std::filesystem::exists(stampInfoPath)) {
                if (m_inCacheFrames.empty()) {//重新运行了
                    if (!assetsInitialized || runtype == "LoadAsset") {//
                        bool ret = fromDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, runtype);
                        if (!ret)
                            return nullptr;
                        for (auto& [k, obj] : m_frames[frameIdx].view_objects.m_curr) {
                            obj->userData().set2("stamp-change", "TotalChange");
                        }
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
                        if (currentFrameNumber != frameid) {//运行时不在起始帧,发生了切帧
                            runtype = "RunAll";
                        }
                        bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype);
                        if (!ret)
                            return nullptr;
                    }
                } else {//切帧
                    for (auto& [objPtrId, flag] : sceneLoadedFlag) {
                        std::get<2>(flag) = true;
                    }
                    runtype = "RunAll"; //切帧要加载新帧的全部
                    bool ret = fromDiskByStampinfo(cacheFramePath, frameid, m_frames[frameIdx].view_objects, baseframeinfo, runtype);
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
            if (std::filesystem::is_directory(entry.path()) || filePath.substr(filePath.size() - 9) != ".zencache")
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
    if (frame == endFrameNumber && std::filesystem::exists(std::filesystem::u8path(cacheFramePath)) && std::filesystem::is_empty(std::filesystem::u8path(cacheFramePath)))
    {
        std::filesystem::remove(std::filesystem::u8path(cacheFramePath));
        zeno::log_info("remove dir: {}", std::filesystem::u8path(cacheFramePath).string());
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

}
