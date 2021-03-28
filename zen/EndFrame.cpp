#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <Hg/IPC/SharedMemory.hpp>
#include <Hg/IPC/Socket.hpp>
#include <cstring>
#include <vector>

namespace zenbase {

struct EndFrame : zen::INode {
  virtual void apply() override {
    Socket sock("/tmp/zenipc/command");

    dprintf(sock.filedesc(), "@ENDF 0\n");
  }
};

static int defEndFrame = zen::defNodeClass<EndFrame>("EndFrame",
    { /* inputs: */ {
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
