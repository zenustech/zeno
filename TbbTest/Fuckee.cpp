#include <iostream>

int fucker_main();

int main()
{
  auto msg = new std::string("Fuckee main");
  std::cout << *msg << std::endl;
  delete msg;
  fucker_main();
  msg = new std::string("Fuckee exit");
  std::cout << *msg << std::endl;
  delete msg;
  return 0;
}
