#include <openvdb/openvdb.h>
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/EventCallbacks.h>

namespace zeno {

static int defOpenvdbInit = getSession().eventCallbacks->hookEvent("init", [] {
    zeno::log_debug("Initializing OpenVDB...");
    openvdb::initialize();
    zeno::log_debug("Initialized OpenVDB successfully!");
});

}
