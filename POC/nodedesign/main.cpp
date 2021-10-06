#include <cstdio>
#include <string>
#include <vector>


struct Node {
    std::string kind;
    int xorder = 0;
    std::vector<int> deps;
    float value = 0;
};


std::vector<Node> nodes(4);


/*void init() {
    if a and b have a common trivial output, then should link.

    0 = a
    1 = b
    2 = c 0 1

    0 = a
    1 = b
    2 = c
    3 = d 0 1
    4 = e 1 2
    5 = if 3 4
}*/


void sortexec(std::vector<int> &tolink) {
    std::sort(tolink.begin(), tolink.end(), [&] (int i, int j) {
        return nodes[i].xorder < nodes[j].xorder;
    });
    for (auto idx: tolink) {
        printf("%d\n", idx);
    }
}

float resolve(int idx);

void touch(int idx, std::vector<int> &tolink) {
    if (nodes[idx].kind == "if") {
        if (resolve(nodes[idx].deps[0])) {
            return touch(nodes[idx].deps[1], tolink);
        } else {
            return touch(nodes[idx].deps[2], tolink);
        }
    }
    for (auto dep: nodes[idx].deps) {
        touch(dep, tolink);
    }
    tolink.push_back(idx);
}

float resolve(int idx) {
    std::vector<int> tolink;
    touch(idx, tolink);
    sortexec(tolink);
    return nodes[idx].value;
}

int main() {
    nodes[0] = {"float", 100, {}};
    nodes[1] = {"float", 200, {}};
    nodes[2] = {"float", 300, {}};
    nodes[3] = {"iff", 400, {2, 0, 1}};

    resolve(3);
    return 0;
}
