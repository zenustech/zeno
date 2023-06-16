#include <zeno/zeno.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/log.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/assetDir.h>
#include <filesystem>
#include <fstream>

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
                cachebasedir = "/tmp";
            }
        }
        auto cachefile = std::filesystem::u8path(cachebasedir) / ("CTD-" + myname + ".zenobjbinarycache");
        return cachefile.u8string();
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

}
}
