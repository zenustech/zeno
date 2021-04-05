#include <iostream>

int main()
{
  auto msg = new std::string("Fuckee main");
  std::cout << *msg << std::endl;
  delete msg;
  return 0;
}
