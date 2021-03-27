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


struct Server {
  Socket::Server serv{"/tmp/zenipc/command"};

  std::map<int, std::unique_ptr<FrameData>> frames;
  int frameid = 1;

  void fetchcmd() {
    Socket sock;
    serv.set_nonblock(true);
    bool ready = serv.listen(&sock);
    serv.set_nonblock(false);
    if (!ready)
      return;

    char buf[1024];
    size_t num = sock.read(buf, sizeof(buf));
    buf[num] = 0;
    printf("COMMAND: %s\n", buf);

    size_t memsize = 0;
    char type[32] = {0};
    sscanf(buf, "@%s%zd", type, &memsize);
    if (!strcmp(type, "ENDF")) {
      frameid++;
      return;
    }

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
};


int main() {
  Server serv;
  while (true) {
    serv.fetchcmd();
    ::usleep(5000);  // sleep for 5 ms
  }
  return 0;
}
