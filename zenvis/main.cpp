#include <Hg/IPC/Socket.hpp>
#include <Hg/IPC/SharedMemory.hpp>
#include <cstring>
#include <vector>
#include <memory>
#include <string>
#include <map>


struct ObjectData {
  std::unique_ptr<std::vector<char>> serial;
  std::string type;
};


struct FrameData {
  std::vector<std::unique_ptr<ObjectData>> objects;
};


std::map<int, std::unique_ptr<FrameData>> frames;
int frameid = 1;


void server_loop() {
  Socket::Server serv("/tmp/zenipc/command");

  while (true) {
    Socket sock = serv.listen();

    char buf[1024];
    size_t num = sock.read(buf, sizeof(buf));
    buf[num] = 0;
    printf("COMMAND: %s\n", buf);

    size_t memsize = 0;
    char type[32] = {0};
    sscanf(buf, "@%s%zd", type, &memsize);
    SharedMemory shm("/tmp/zenipc/memory", memsize);

    if (frames.find(frameid) == frames.end()) {
      frames[frameid] = std::make_unique<FrameData>();
    }
    auto const &frm = frames.at(frameid);

    auto obj = std::make_unique<ObjectData>();
    obj->serial = std::make_unique<std::vector<char>>(shm.size());
    std::memcpy(obj->serial->data(), shm.data(), shm.size());
    obj->type = std::string(type);
    frm->objects.push_back(std::move(obj));
  }
}


int main(int argc, char **argv)
{
  server_loop();
  return 0;
}
