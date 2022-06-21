#include <zeno/extra/EventCallbacks.h>
#include <zeno/core/Session.h>
#include "zstartup.h"

void startUp()
{
    static int initOnce = (zeno::getSession().eventCallbacks->triggerEvent("init"), 0);
}
