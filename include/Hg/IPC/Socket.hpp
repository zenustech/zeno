#pragma once

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>


class Socket {
  void create() {
    conn = ::socket(PF_UNIX, SOCK_STREAM, 0);
    if (conn < 0) {
      ::perror("socket");
      return;
    }
  }

  void open(const char *domain) {
    struct sockaddr_un srv_addr;
    srv_addr.sun_family = AF_UNIX;
    ::strcpy(srv_addr.sun_path, domain);

    int ret = ::connect(conn, (struct sockaddr *)&srv_addr, sizeof(srv_addr));
    if (ret < 0) {
      ::perror(domain);
      return;
    }
  }

protected:
  int conn{-1};

public:
  void close() {
    if (conn >= 0) {
      ::close(conn);
      conn = -1;
    }
  }

  void write(const void *buf, size_t size) {
    ::write(conn, buf, size);
  }

  size_t read(void *buf, size_t size) {
    return ::read(conn, buf, size);
  }

  explicit Socket(int conn = -1) : conn(conn) {}

  explicit Socket(const char *domain) {
    this->create();
    this->open(domain);
  }

  ~Socket() {
    this->close();
  }
};


class SocketServer {
  int conn{-1};

  void create() {
    conn = ::socket(PF_UNIX, SOCK_STREAM, 0);
    if (conn < 0) {
      ::perror("socket");
      return;
    }
  }

  void close() {
    if (conn >= 0) {
      ::close(conn);
      conn = -1;
    }
  }

  void open(const char *domain) {
    struct sockaddr_un srv_addr;
    srv_addr.sun_family = AF_UNIX;
    ::strcpy(srv_addr.sun_path, domain);
    ::unlink(domain);

    int ret = ::bind(conn, (struct sockaddr *)&srv_addr, sizeof(srv_addr));
    if (ret < 0) {
      ::perror(domain);
      return;
    }
  }

public:
  Socket listen(int backlog = 1) {
    int ret = ::listen(conn, backlog);
    if (ret < 0) {
      ::perror("listen");
      return Socket();
    }

    struct sockaddr_un clt_addr;
    socklen_t len = sizeof(clt_addr);
    int fd = ::accept(conn, (struct sockaddr *)&clt_addr, &len);
    if (fd < 0) {
      ::perror("accept");
      return Socket();
    }
    return Socket(fd);
  }

  explicit SocketServer(const char *domain) {
    this->create();
    this->open(domain);
  }

  ~SocketServer() {
    this->close();
  }

  void write(const void *buf, size_t size) {
    ::write(conn, buf, size);
  }

  size_t read(void *buf, size_t size) {
    return ::read(conn, buf, size);
  }
};
