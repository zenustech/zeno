#include <iostream>
#include <memory>

int main() {
    auto x = std::make_unique<int>(42);
    std::cout << *x << std::endl;
    return 0;
}
