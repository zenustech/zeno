#pragma once

#include <zeno/utils/envconfig.h>

#include <string>
#include <iostream>

struct SyncInfo{
    std::string sMsg;
};

struct HandleStateInfo{
    std::string cPath;

    std::function<void()> cUpdateFunction;
    std::function<void(SyncInfo)> cSyncCallback;
    int cPort;

    HandleStateInfo(){
        cPath = zeno::envconfig::get("UsdRepoPath");
        cPort = zeno::envconfig::getInt("ServerPort");
        std::cout << "USD: State Info Path " << cPath << " Port " << cPort << "\n";
    }
};