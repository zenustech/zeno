#include <zeno/extra/EventCallbacks.h>
#include <zeno/core/Session.h>
#include "zstartup.h"

void startUp()
{
    zeno::getSession().eventCallbacks->triggerEvent("init");
}
