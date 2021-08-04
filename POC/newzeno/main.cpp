#define FMT_HEADER_ONLY
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <any>

struct Method {
    struct ArgInfo {
        std::string type;
        std::string name;
    };

    std::string name;
    std::vector<ArgInfo> arguments;
    std::vector<ArgInfo> returns;
};

struct Node {
    int type;
};

struct Link {
    int src_node, dst_node;
    int src_socket, dst_socket;
};

std::vector<Method> methods;
std::vector<Node> nodes;
std::vector<Link> links;

void gen_method_declare(Method const &method) {
    std::cout << "std::tuple\n";
    for (int a = 0; a < (int)method.returns.size(); a++) {
        std::cout << (a == 0 ? "< " : ", ");
        std::cout << method.arguments[a].type;
        std::cout << " ";
        std::cout << method.arguments[a].name;
        std::cout << "\n";
    }
    std::cout << method.name << "\n";
    for (int a = 0; a < (int)method.arguments.size(); a++) {
        std::cout << (a == 0 ? "( " : ", ");
        std::cout << method.arguments[a].type;
        std::cout << " ";
        std::cout << method.arguments[a].name;
        std::cout << "\n";
    }
    std::cout << (method.arguments.size() ? ")" : "()") << ";\n";
}

void codegen() {
    for (int i = 0; i < (int)methods.size(); i++) {
        auto const &method = methods[i];
        gen_method_declare(method);
    }
}

int main() {
}
