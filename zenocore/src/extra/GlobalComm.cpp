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
#ifdef __linux__
    #include <unistd.h>
    #include <sys/statfs.h>
#endif
#include <chrono>
#include <thread>
#define MIN_DISKSPACE_MB 1024

namespace zeno {

std::vector<std::filesystem::path> cachepath(3);
std::unordered_set<std::string> lightCameraNodes({
    "CameraEval", "CameraNode", "CihouMayaCameraFov", "ExtractCameraData", "GetAlembicCamera","MakeCamera",
    "LightNode", "BindLight", "ProceduralSky", "HDRSky", "SkyComposer"
    });
std::set<std::string> matNodeNames = {"ShaderFinalize", "ShaderVolume", "ShaderVolumeHomogeneous"};

void GlobalComm::toDisk(std::string cachedir, int frameid, std::map<std::string, zany>& objs, bool cacheLightCameraOnly, bool cacheMaterialOnly, std::string fileName) {
    if (cachedir.empty()) return;
    std::filesystem::path dir = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid).substr(1));
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir))
    {
        log_critical("can not create path: {}", dir);
    }
    std::vector<std::vector<char>> bufCaches(3);
    std::vector<std::vector<size_t>> poses(3);
    std::vector<std::string> keys(3);
    for (auto const &[key, obj]: objs) {

        size_t bufsize =0;
        std::string nodeName = key.substr(key.find("-") + 1, key.find(":") - key.find("-") -1);
        if (cacheLightCameraOnly && (lightCameraNodes.count(nodeName) || obj->userData()->get_int("isL") || dynamic_cast<CameraObject*>(obj.get())))
        {
            bufsize = bufCaches[0].size();
            if (encodeObject(obj.get(), bufCaches[0]))
            {
                keys[0].push_back('\a');
                keys[0].append(key);
                poses[0].push_back(bufsize);
            }
        }
        if (cacheMaterialOnly && (matNodeNames.count(nodeName)>0 || dynamic_cast<MaterialObject*>(obj.get())))
        {
            bufsize = bufCaches[1].size();
            if (encodeObject(obj.get(), bufCaches[1]))
            {
                keys[1].push_back('\a');
                keys[1].append(key);
                poses[1].push_back(bufsize);
            }
        }
        if (!cacheLightCameraOnly && !cacheMaterialOnly)
        {
            if (lightCameraNodes.count(nodeName) || obj->userData()->get_int("isL") || dynamic_cast<CameraObject*>(obj.get())) {
                bufsize = bufCaches[0].size();
                if (encodeObject(obj.get(), bufCaches[0]))
                {
                    keys[0].push_back('\a');
                    keys[0].append(key);
                    poses[0].push_back(bufsize);
                }
            } else if (matNodeNames.count(nodeName)>0 || dynamic_cast<MaterialObject*>(obj.get())) {
                bufsize = bufCaches[1].size();
                if (encodeObject(obj.get(), bufCaches[1]))
                {
                    keys[1].push_back('\a');
                    keys[1].append(key);
                    poses[1].push_back(bufsize);
                }
            } else {
                bufsize = bufCaches[2].size();
                if (encodeObject(obj.get(), bufCaches[2]))
                {
                    keys[2].push_back('\a');
                    keys[2].append(key);
                    poses[2].push_back(bufsize);
                }
            }
        }
    }

    if (fileName == "")
    {
        cachepath[0] = dir / "lightCameraObj.zencache";
        cachepath[1] = dir / "materialObj.zencache";
        cachepath[2] = dir / "normalObj.zencache";
    }
    else
    {
        cachepath[2] = std::filesystem::u8path(dir.string() + "/" + fileName);
    }
    size_t currentFrameSize = 0;
    for (int i = 0; i < 3; i++)
    {
        if (poses[i].size() == 0 && (cacheLightCameraOnly && i != 0 || cacheMaterialOnly && i != 1 || fileName != "" && i != 2))
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
        // #ifdef __linux__
        //     zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).string());
        //     ::sleep(2);
        //     statfs(std::filesystem::u8path(cachedir).c_str(), &diskInfo);
        //     freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;

        // #else
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).root_path().string());
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
        // #endif
    }
    for (int i = 0; i < 3; i++)
    {
        if (poses[i].size() == 0 && (cacheLightCameraOnly && i != 0 || cacheMaterialOnly && i != 1 || fileName != "" && i != 2))
            continue;
        log_debug("dump cache to disk {}", cachepath[i]);
        std::ofstream ofs(cachepath[i], std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(keys[i].begin(), keys[i].end(), oit);
        std::copy_n((const char *)poses[i].data(), poses[i].size() * sizeof(size_t), oit);
        std::copy(bufCaches[i].begin(), bufCaches[i].end(), oit);
    }
    objs.clear();
}

bool GlobalComm::fromDisk(std::string cachedir, int frameid, std::map<std::string, zany>& objs, std::string fileName) {
    if (cachedir.empty())
        return false;
    objs.clear();
    auto dir = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1);
    if (fileName == "")
    {
        cachepath[0] = dir / "lightCameraObj.zencache";
        cachepath[1] = dir / "materialObj.zencache";
        cachepath[2] = dir / "normalObj.zencache";
    }
    else
    {
        cachepath[2] = std::filesystem::u8path(dir.string() + "/" + fileName);
    }

    for (auto path : cachepath)
    {
        if (!std::filesystem::exists(path))
        {
            continue;
        }
        log_debug("load cache from disk {}", path);

        auto szBuffer = std::filesystem::file_size(path);
        std::vector<char> dat(szBuffer);
        FILE *fp = fopen(path.string().c_str(), "rb");
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
        std::copy_n(dat.data() + pos, (keyscount + 1) * sizeof(size_t), (char *)poses.data());
        pos += (keyscount + 1) * sizeof(size_t);
        for (int k = 0; k < keyscount; k++) {
            if (poses[k] > dat.size() - pos || poses[k + 1] < poses[k]) {
                log_error("zeno cache file broken (4.{})", k);
            }
            const char *p = dat.data() + pos + poses[k];
            objs.try_emplace(keys[k], decodeObject(p, poses[k + 1] - poses[k]));
        }
    }
    return true;
}

}

