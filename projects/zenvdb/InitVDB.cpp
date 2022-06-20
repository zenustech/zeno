#include <openvdb/openvdb.h>
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/Initializer.h>

namespace zeno {

static int defOpenvdbInit = getSession().initializer->defInit([] {
    zeno::log_debug("Initializing OpenVDB...");
    openvdb::initialize();
    zeno::log_debug("Initialized OpenVDB successfully!");
});

}
