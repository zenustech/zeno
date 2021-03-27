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

  void load(const char *path, size_t size, size_t offset)
  {
    int fd = ::open(path, O_RDWR, 00777);
    if (fd < 0) {
      ::perror(path);
      return;
    }
    m_size = size;
    m_base = ::mmap(NULL, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (!m_base)
      ::perror(path);
    ::close(fd);
  }

public:
  void release()
  {
    if (m_base) {
      ::munmap(m_base, m_size);
      m_base = nullptr;
      m_size = 0;
    }
  }

  SharedMemory(const char *path, size_t size, size_t offset = 0)
  {
    this->load(path, size, offset);
  }

  void *data() const
  {
    return m_base;
  }

  size_t size() const
  {
    return m_size;
  }

  ~SharedMemory()
  {
    this->release();
  }
};
