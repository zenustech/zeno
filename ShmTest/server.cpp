#include <Hg/IPC/Socket.hpp>


int main(int argc, char** argv)
{
  SocketServer serv("/tmp/UNIX.domain");

  Socket sock = serv.listen();
  char buf[1024];
  size_t num = sock.read(buf, sizeof(buf));
  printf("recv: %d %s\n", num, buf);
 
  return 0;
}  
