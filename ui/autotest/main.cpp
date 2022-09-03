#include <zenoapi/include/zenoapi.h>

int main()
{
    zenoapi::openFile("E:/zeno/cube.zsg");
    zenoapi::addNode("main", "CreateTube");
    return 0;
}