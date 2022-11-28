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

    HandleStateInfo(){
        cPath = zeno::envconfig::get("UsdRepoPath");

        std::cout << "USD: State Info Path " << cPath << "\n";
    }
};