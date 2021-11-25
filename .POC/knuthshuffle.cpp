#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>

void shuffle(std::vector<int> &D) {
    for (int i = 0; i < (int)D.size(); i++) {
        float r = drand48();
        int p = int(r * i) + 1;
        std::swap(D[p], D[i]);
    }
}

int main() {
    std::vector<int> a;
    for (int i = 0; i < 10; i++) {
        a.push_back(i);
    }
    shuffle(a);
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << std::endl;
    }
    return 0;
}
