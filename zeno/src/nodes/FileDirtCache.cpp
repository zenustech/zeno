#include <zeno/zeno.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/log.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/GlobalComm.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <cstdlib>

namespace zeno {
namespace {

struct CacheToDisk : zeno::INode {
    virtual void preApply() override {
        if (auto it = inputBounds.find("object"); it != inputBounds.end()) {
            auto snid = it->second.first;
            auto &dc = graph->getDirtyChecker();
            if (dc.amIDirty(snid)) {
                invalidateCache();
            } else {
                if (auto cached = tryGetCached()) {
                    log_info("CacheToDisk: reusing cache at {}", getCachePath());
                    set_output("object", std::move(cached));
                    return;
                }
            }
        } else {
            throw makeError("CacheToDisk: input socket object not connected");
        }
        log_info("CacheToDisk: updating cache at {}", getCachePath());
        INode::preApply();
    }

    virtual void apply() override {
        auto obj = get_input("object");
        if (obj) {
            std::vector<char> out;
            encodeObject(obj.get(), out);
            auto cachefile = getCachePath();
            if (std::ofstream ofs(cachefile); !ofs) {
                log_error("failed to open file for write: {}", cachefile);
            } else {
                std::ostreambuf_iterator<char> oit(ofs);
                std::copy(out.begin(), out.end(), oit);
            }
        }
        set_output("object", std::move(obj));
    }

    void invalidateCache() {
        auto cachefile = getCachePath();
        if (std::filesystem::exists(cachefile)) {
            std::filesystem::remove(cachefile);
            if (std::filesystem::exists(cachefile)) {
                throw makeError(format("failed to remove out-of-date cache file: {}", cachefile));
            }
        }
    }

    std::string getCachePath() {
        auto cachebasedir = get_param<std::string>("cachebasedir");
        if (cachebasedir.empty()) {
            cachebasedir = zeno::getConfigVariable("ZENCACHE");
            if (cachebasedir.empty()) {
                cachebasedir = std::filesystem::temp_directory_path().string();
            }
        }
        auto cachefile = std::filesystem::u8path(cachebasedir) / ("CTD-" + myname + ".zenobjbinarycache");
        return cachefile.string();
    }

    std::shared_ptr<IObject> tryGetCached() {
        auto cachefile = getCachePath();
        if (!std::filesystem::exists(cachefile)) {
            return nullptr;
        }
        std::ifstream ifs(cachefile);
        if (!ifs) {
            log_error("failed to open file for read: {}", cachefile);
            return nullptr;
        }
        std::istreambuf_iterator<char> iit(ifs), eiit;
        std::vector<char> dat;
        std::copy(iit, eiit, std::back_inserter(dat));
        auto obj = decodeObject(dat.data(), dat.size());
        if (!obj) {
            log_error("failed to decode object in file: {}", cachefile);
            return nullptr;
        }
        return obj;
    }
};

ZENO_DEFNODE(CacheToDisk)({
    {
       {"object"},
    },
    {
       {"object"},
    },
    {
       {"string", "cachebasedir", ""},
    },
    {"lifecycle"},
});

struct EmbedZsgGraph : zeno::INode {
    virtual void apply() override {
        auto zsgPath = get_input2<std::string>("zsgPath");
        auto zsgp = std::filesystem::u8path(zsgPath).string();
        auto zslPath = (std::filesystem::temp_directory_path() / (".tmpEZG-" + std::to_string(std::random_device()()) + "-tmp.zsl")).string();
        auto zslp = std::filesystem::u8path(zslPath).string();
        auto cmd = zeno::getConfigVariable("EXECFILE") + " -invoke dumpzsg2zsl " + zsgp + " " + zslp;
        log_info("executing command: [{}]", cmd);
        std::system(cmd.c_str());
        log_info("done execution, zsl should be generated");
        std::string content;
        if (std::ifstream ifs(zslPath); !ifs) {
            throw makeError("failed to generate temporary zsl file!\n");
        } else {
            std::istreambuf_iterator<char> iit(ifs), eiit;
            std::copy(iit, eiit, std::back_inserter(content));
            ifs.close();
            /* std::filesystem::remove(zslPath); */
        }
        auto g = getThisSession()->createGraph();
        g->addSubnetNode("custom")->loadGraph(content.c_str());
        auto argsDict = has_input("argsDict") ? get_input<DictObject>("argsDict") : std::make_shared<DictObject>();
        auto retsDict = std::make_shared<DictObject>();
        retsDict->lut = g->callSubnetNode("custom", argsDict->lut);
        /* for (auto &[k, v]: retsDict->lut) { */
        /*     zeno::log_warn("mama [{}] [{}]", k, v); */
        /* } */
        set_output("retsDict", std::move(retsDict));
    }
};

ZENO_DEFNODE(EmbedZsgGraph)({
    {
       {"readpath", "zsgPath", ""},
       {"dict", "argsDict"},
    },
    {
       {"dict", "retsDict"},
    },
    {
    },
    {"subgraph"},
});

struct ReadCacheFromDir : zeno::INode {
    virtual void apply() override {
        auto cachePath = get_input2<std::string>("cachePath");
        auto frame = get_input2<std::string>("frame");

        // 确保路径存在
        if (!std::filesystem::exists(cachePath)) {
            throw makeError(format("Cache path does not exist: {}", cachePath));
        }

        // 检查是否是目录
        if (!std::filesystem::is_directory(cachePath)) {
            throw makeError(format("Cache path is not a directory: {}", cachePath));
        }

        // 如果frame参数为空，输出所有帧的结果
        if (frame.empty()) {
            // 创建主DictObject，用于存储所有帧的缓存
            auto mainDict = std::make_shared<zeno::DictObject>();

            // 遍历cachePath下的所有目录
            for (const auto& entry : std::filesystem::directory_iterator(cachePath)) {
                if (entry.is_directory()) {
                    std::string dirName = entry.path().filename().string();

                    // 确定帧号和加载模式
                    int frameNum = 0;
                    bool loadasset = false, switchTimeline = false;;

                    if (dirName == "data") {
                        // data目录，使用asset模式
                        loadasset = true;
                    } else {
                        // 尝试将目录名解析为帧号
                        switchTimeline = true;
                        try {
                            frameNum = std::stoi(dirName);
                        } catch (const std::exception& e) {
                            // 如果不是数字目录，跳过
                            continue;
                        }
                    }

                    // 使用GlobalComm::fromDiskByStampinfo读取该目录的cache
                    zeno::GlobalComm globalComm;
                    zeno::GlobalComm::ViewObjects objs;
                    std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>> newFrameStampInfo;
                    std::string runtype = "RunAll";

                    bool success = globalComm.fromDiskByStampinfo(cachePath, frameNum, objs, newFrameStampInfo, runtype, switchTimeline, loadasset);

                    if (!success) {
                        log_warn("ReadCacheFromDir: failed to load cache from directory {}", dirName);
                        continue;
                    }

                    // 创建该帧的DictObject
                    auto frameDict = std::make_shared<zeno::DictObject>();

                    // 遍历objs.m_curr中的所有对象，添加到frameDict中
                    for (auto& [key, obj] : objs.m_curr) {
                        if (obj) {
                            frameDict->lut[key] = std::move(obj);
                        }
                    }

                    if (!frameDict->lut.empty()) {
                        // 确定dict的key：如果是data目录就用"data"，否则用帧号的数字字符串
                        std::string dictKey = loadasset ? "data" : std::to_string(frameNum);
                        mainDict->lut[dictKey] = std::move(frameDict);
                    } else {
                        log_warn("ReadCacheFromDir: no objects found in directory {}", dirName);
                    }
                }
            }

            if (mainDict->lut.empty()) {
                log_warn("ReadCacheFromDir: no valid cache directories found in {}", cachePath);
            }

            set_output("dict", std::move(mainDict));
        } else {
            // frame参数不为空，输出指定帧的结果
            int frameNum = 0;
            bool loadasset = false, switchTimeline = false;

            // 解析frame参数
            if (frame == "-1") {
                // 负数为data目录
                loadasset = true;
            } else {
                switchTimeline = true;
                try {
                    frameNum = std::stoi(frame);
                } catch (const std::exception& e) {
                    throw makeError(format("Invalid frame number: {}", frame));
                }

                auto dir = std::filesystem::u8path(cachePath) / (loadasset ? "data" : std::to_string(1000000 + frameNum).substr(1));
                if (!std::filesystem::exists(dir)) {
                    throw makeError(format("Invalid frame number: {}", frame));
                }
            }

            // 使用GlobalComm::fromDiskByStampinfo读取指定帧的cache
            zeno::GlobalComm globalComm;
            zeno::GlobalComm::ViewObjects objs;
            std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>> newFrameStampInfo;
            std::string runtype = "RunAll";

            bool success = globalComm.fromDiskByStampinfo(cachePath, frameNum, objs, newFrameStampInfo, runtype, switchTimeline, loadasset);

            if (!success) {
                throw makeError(format("Failed to load cache for frame {} from directory {}", frame, cachePath));
            }

            // 创建该帧的DictObject
            auto frameDict = std::make_shared<zeno::DictObject>();

            frameDict->lut = std::move(objs.m_curr);

            if (frameDict->lut.empty()) {
                log_warn("ReadCacheFromDir: no objects found for frame {} in directory {}", frame, cachePath);
            }

            set_output("dict", std::move(frameDict));
        }
    }
};

ZENO_DEFNODE(ReadCacheFromDir)({
    {
       {"directory", "cachePath", ""},
       {"string", "frame", "-1"},
    },
    {
       {"dict", "dict"},
    },
    {
    },
    {"file"},
});

}
}
