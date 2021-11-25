#include <iostream>
#include <memory>

int main() {
    auto x = new int(42);
    std::cout << *x << std::endl;
    return 0;
}
