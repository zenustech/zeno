#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>


int main(int argc, char** argv)  
{  
  int conn = socket(PF_UNIX, SOCK_STREAM, 0);
  if (conn < 0) {
    perror("socket");
    return 1;
  }

  const char *domain = "/tmp/UNIX.domain";

  struct sockaddr_un srv_addr;
  srv_addr.sun_family = AF_UNIX;
  strcpy(srv_addr.sun_path, domain);

  int ret = connect(conn, (struct sockaddr *)&srv_addr, sizeof(srv_addr));
  if (ret < 0) {
    perror(domain);
    close(conn);
    return 1;
  }

  char buf[] = "message from client";

  write(conn, buf, sizeof(buf));

  close(conn);
  return 0;  
}  
