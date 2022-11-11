#pragma once

#include <string>

struct SyncInfo{
    std::string sMsg;
};

struct HandleStateInfo{
    std::string cRepo = "http://test1:12345@192.168.2.106:8000/r/zeno_usd_test.git";
    std::string cPath = "C:/Users/Public/zeno_usd_test";
    std::string cServer = "192.168.2.106";

    std::function<void()> cUpdateFunction;
    std::function<void(SyncInfo)> cSyncCallback;
};