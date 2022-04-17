#include <z2/dop/dop.h>
#include <z2/ds/Mesh.h>
#include <string_view>
#include <sstream>
#include <fstream>
#include <cstring>
#include <tuple>


namespace z2::ds {
namespace {


inline ztd::vec3f read_vec3f(std::string_view const &exp) {
    int i = 0;
    ztd::vec3f tokens(0);
    std::string token;
    std::istringstream iss((std::string)exp);
    for (int i = 0; i < 3 && std::getline(iss, token, ' '); i++)
        tokens[i] = std::stof(token);
    return tokens;
}


inline ztd::vec2f read_vec2f(std::string_view const &exp) {
    int i = 0;
    ztd::vec2f tokens(0);
    std::string token;
    std::istringstream iss((std::string)exp);
    for (int i = 0; i < 2 && std::getline(iss, token, ' '); i++)
        tokens[i] = std::stof(token);
    return tokens;
}


inline std::tuple<int, int, int> read_tuple3i(std::string_view const &exp) {
    int i = 0, tokens[3] = {0, 0, 0};
    std::string token;
    std::istringstream iss((std::string)exp);
    for (int i = 0; i < 3 && std::getline(iss, token, '/'); i++)
        tokens[i] = std::stoi(token);
    return {tokens[0], tokens[1], tokens[2]};
}


static void readMeshFromOBJ(std::istream &in, Mesh &mesh) {
    std::vector<ztd::vec2f> uv_vert;
    std::vector<int> uv_loop;

    char buf[1025];
    while (in.getline(buf, 1024, '\n')) {
        std::string_view line = buf;
        if (line.ends_with("\n")) {
            line = line.substr(0, line.size() - 1);
        }
        if (line.ends_with("\r")) {
            line = line.substr(0, line.size() - 1);
        }

        if (line.starts_with("v ")) {
            mesh.vert.emplace_back(read_vec3f(line.substr(2)));

        } else if (line.starts_with("vt ")) {
            uv_vert.emplace_back(read_vec2f(line.substr(3)));

        } else if (line.starts_with("f ")) {
            line = line.substr(2);

            int start = mesh.loop.size(), num = 0;
            while (num++ < 4096) {
                auto next = line.find(' ');
                auto [v, vt, vn] = read_tuple3i(line.substr(0, next));

                mesh.loop.push_back(v - 1);
                if (vt != 0)
                    uv_loop.push_back(vt - 1);

                if (next == std::string::npos)
                    break;
                line = line.substr(next + 1);
            }
            mesh.poly.emplace_back(start, num);
        }
    }

    mesh.loop_uv.reserve(uv_loop.size());
    for (int i = 0; i < uv_loop.size(); i++) {
        mesh.loop_uv.push_back(uv_vert.at(uv_loop[i]));
    }
}


struct ReadOBJMesh : dop::Node {
    void apply() override {
        auto path = get_input<std::string>(0);
        auto mesh = std::make_shared<Mesh>();
        std::ifstream fin(path);
        readMeshFromOBJ(fin, *mesh);
        set_output(0, mesh);
    }
};

// Z2_DOP_DEFINE(ReadOBJMesh, {{
//     "mesh", "load mesh from .obj file",
// }, {
//     {"path"},
// }, {
//     {"mesh"},
// }});


}
}
