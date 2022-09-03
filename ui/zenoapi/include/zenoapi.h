#ifndef __ZENO_UI_API_H__
#define __ZENO_UI_API_H__

#include "interface.h"

namespace zenoapi
{
    void openFile(const std::string& fn);
    std::string addNode(const std::string& subg, const std::string& nodeCls);
}


#endif