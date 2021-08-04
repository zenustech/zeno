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

struct Invoke {
    int method;
    std::vector<std::pair<int, int>> arguments;
};

std::vector<Method> methods;
std::vector<Invoke> invokes;

void gen_method_declare(Method const &method) {
    std::cout << "std::tuple\n";
    for (int a = 0; a < (int)method.returns.size(); a++) {
        std::cout << (a == 0 ? "< " : ", ");
        std::cout << method.arguments[a].type;
        std::cout << " ";
        std::cout << method.arguments[a].name;
        std::cout << "\n";
    }
    std::cout << (method.returns.size() ? ">" : "<>") << "\n";
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
    methods.emplace_back();
    methods.back().arguments.resize(2);
    methods.back().returns.resize(1);
    methods.back().arguments[0].type = "int";
    methods.back().arguments[0].name = "lhs";
    methods.back().arguments[1].type = "int";
    methods.back().arguments[1].name = "rhs";
    methods.back().returns[0].type = "int";
    methods.back().returns[0].name = "ret";
    methods.back().name = "my_add";
    codegen();
}
