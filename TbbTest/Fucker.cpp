#include "tbb/tbbmalloc_proxy.h"
#include "tbb/scalable_allocator.h"
#include <iostream>

int fucker_main()
{
  auto msg = new std::string("Fucker load");
  std::cout << *msg << std::endl;
  delete msg;
  return 0;
}
