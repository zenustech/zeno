#include "startup/zstartup.h"
#include <Windows.h>


int main(int argc, char* argv[])
{
    //MessageBoxA(0, "launch", "", MB_OK);
    if (argc >= 7 && !strcmp(argv[1], "-runner")) {
        //MessageBoxA(0, "ABC", "abc", MB_OK);
        extern int runner_main(int sessionid, int port, char* path);
        int sessionid = atoi(argv[2]);
        int port = atoi(argv[4]);
        char* path = argv[6];
        return runner_main(sessionid, port, path);
    }
}