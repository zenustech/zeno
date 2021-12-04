#include <zeno/zty/mesh/MeshIO.h>
#include <zeno/zty/mesh/Mesh.h>
#include <string_view>
#include <sstream>
#include <tuple>


ZENO_NAMESPACE_BEGIN
namespace zty {


static math::vec3f read_vec3f(std::string_view const &exp) {
    int i = 0;
    math::vec3f tokens(0);
    std::string token;
    std::istringstream iss((std::string)exp);
    for (int i = 0; i < 3 && std::getline(iss, token, ' '); i++)
        tokens[i] = std::stof(token);
    return tokens;
}


static math::vec2f read_vec2f(std::string_view const &exp) {
    int i = 0;
    math::vec2f tokens(0);
    std::string token;
    std::istringstream iss((std::string)exp);
    for (int i = 0; i < 2 && std::getline(iss, token, ' '); i++)
        tokens[i] = std::stof(token);
    return tokens;
}


static std::tuple<int, int, int> read_tuple3i(std::string_view const &exp) {
    int i = 0, tokens[3] = {0, 0, 0};
    std::string token;
    std::istringstream iss((std::string)exp);
    for (int i = 0; i < 3 && std::getline(iss, token, '/'); i++)
        tokens[i] = std::stoi(token);
    return {tokens[0], tokens[1], tokens[2]};
}


void readMeshFromOBJ(std::istream &in, Mesh &mesh) {
    std::vector<math::vec2f> uv_vert;
    std::vector<int> uv_loop;
    auto &vert = mesh.vert;
    auto &loop = mesh.loop;
    auto &poly = mesh.poly;
    auto &loop_uv = mesh.loop_uv;

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
            vert.emplace_back(read_vec3f(line.substr(2)));

        } else if (line.starts_with("vt ")) {
            uv_vert.emplace_back(read_vec2f(line.substr(3)));

        } else if (line.starts_with("f ")) {
            line = line.substr(2);

            uint32_t num = 0;
            while (num++ < 4096) {
                auto next = line.find(' ');
                auto [v, vt, vn] = read_tuple3i(line.substr(0, next));

                loop.push_back(v - 1);
                if (vt != 0)
                    uv_loop.push_back(vt - 1);

                if (next == std::string::npos)
                    break;
                line = line.substr(next + 1);
            }
            poly.push_back(num);
        }
    }

    loop_uv.reserve(uv_loop.size());
    for (int i = 0; i < uv_loop.size(); i++) {
        loop_uv.push_back(uv_vert.at(uv_loop[i]));
    }
}


void writeMeshToOBJ(std::ostream &out, Mesh const &mesh) {
    for (auto &v : mesh.vert) {
        out << 'v' << ' ' << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    }

    size_t start = 0;
    for (auto &p : mesh.poly) {
        out << 'f';
        for (size_t l = start; l < start + p; ++l) {
            out << ' ' << mesh.loop[l] + 1;
        }
        out << '\n';
        start += p;
    }
}


}
ZENO_NAMESPACE_END
