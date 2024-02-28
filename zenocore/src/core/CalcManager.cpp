#include <zeno/core/CalcManager.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>


namespace zeno {

    CalcManager::CalcManager() {

    }

    ZENO_API void CalcManager::run() {
        getSession().mainGraph->runGraph();
    }

    ZENO_API void CalcManager::mark_frame_change_dirty()
    {
        getSession().mainGraph->markDirtyWhenFrameChanged();
    }

}