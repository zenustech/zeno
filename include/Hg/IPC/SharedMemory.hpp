#pragma once

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>


class SharedMemory {
  void *m_base{nullptr};
  size_t m_size{0};

  void release()
  {
    if (m_base)
      munmap(m_base, m_size);
    m_base = nullptr;
  }

  void load(const char *path, size_t size, size_t offset)
  {
    int fd = open(path, O_RDWR, 00777);
    if (fd < 0) {
      perror(path);
      return;
    }
    m_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (!m_base)
      perror(path);
    close(fd);
  }

public:
  SharedMemory(const char *path, size_t size, size_t offset = 0)
  {
    load(path, size, offset);
  }

  void *data() const {
    return m_base;
  }

  size_t size() const {
    return m_size;
  }

  ~SharedMemory()
  {
    release();
  }
};
