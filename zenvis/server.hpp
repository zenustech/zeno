#pragma once

#include <Hg/IPC/Socket.hpp>
#include <Hg/IPC/SharedMemory.hpp>
#include "frames.hpp"
#include <fcntl.h>


namespace zenvis {


struct Server {
  int _mkdir = ::mkdir("/tmp/zenipc", 0777);
  Socket::Server serv{"/tmp/zenipc/command"};
  Socket::Server init_serv{"/tmp/zenipc/initialize"};
  int frameid = 0;

  void poll_init() {
    Socket sock;
    init_serv.set_nonblock(true);
    bool ready = init_serv.listen(&sock);
    init_serv.set_nonblock(false);
    if (!ready)
      return;
    printf("=== INITIALIZE ===\n\n");

    sock.readchar();
    frameid = 0;
    frames.clear();
    sock.writechar('%');
  }

  bool poll_once() {
    Socket sock;
    serv.set_nonblock(true);
    bool ready = serv.listen(&sock);
    serv.set_nonblock(false);
    if (!ready)
      return false;

    char buf[1024];
    size_t num = sock.read(buf, sizeof(buf));
    buf[num] = 0;
    printf("COMMAND: %s\n", buf);

    size_t memory_size = 0, shader_size = 0;
    char type[32] = {0};
    sscanf(buf, "@%30s%zd%zd", type, &memory_size, &shader_size);

    if (!strcmp(type, "ENDF")) {
      frameid++;
      sock.writechar('%');
      return false;
    }

    SharedMemory shm_memory("/tmp/zenipc/memory", memory_size);
    SharedMemory shm_shader("/tmp/zenipc/shader", shader_size);

    if (frames.find(frameid) == frames.end()) {
      frames[frameid] = std::make_unique<FrameData>();
    }
    auto const &frm = frames.at(frameid);

    auto obj = std::make_unique<ObjectData>();
    obj->type = std::string(type);

    obj->memory = std::make_unique<std::vector<char>>(shm_memory.size());
    std::memcpy(obj->memory->data(), shm_memory.data(), shm_memory.size());

    obj->shader = std::make_unique<std::vector<char>>(shm_shader.size());
    std::memcpy(obj->shader->data(), shm_shader.data(), shm_shader.size());

    frm->objects.push_back(std::move(obj));

    sock.writechar('%');
    return true;
  }

  void poll() {
    // keep polling until queue empty or frame ends
    while (poll_once());
  }

private:
  static std::unique_ptr<Server> _instance;
  struct _PrivCtor {};

public:
  Server(_PrivCtor) {}
  static Server &get() {
    if (!_instance)
      _instance = std::make_unique<Server>(_PrivCtor{});
    return *_instance;
  }
};


}
