#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <cstring>

namespace zenbase {

static void readobj(
    const char *path,
    std::vector<glm::vec3> &face_vertices,
    std::vector<glm::vec2> &face_uvs,
    std::vector<glm::vec3> &face_normals)
{
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec3> normals;
  std::vector<unsigned> vertex_indices;
  std::vector<unsigned> uv_indices;
  std::vector<unsigned> normal_indices;

  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror(path);
    abort();
  }

  char hdr[128];
  while (EOF != fscanf(fp, "%s", hdr)) {
    if (!strcmp(hdr, "v")) {
      glm::vec3 vertex;
      fscanf(fp, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
      vertices.push_back(vertex);

    } else if (!strcmp(hdr, "vt")) {
      glm::vec2 uv;
      fscanf(fp, "%f %f\n", &uv.x, &uv.y);
      uvs.push_back(uv);

    } else if (!strcmp(hdr, "vn")) {
      glm::vec3 normal;
      fscanf(fp, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
      normals.push_back(normal);

    } else if (!strcmp(hdr, "f")) {
      glm::uvec3 last_index, first_index, index;

      fscanf(fp, "%d/%d/%d", &index.x, &index.y, &index.z);
      first_index = index;

      fscanf(fp, "%d/%d/%d", &index.x, &index.y, &index.z);
      last_index = index;

      while (fscanf(fp, "%d/%d/%d", &index.x, &index.y, &index.z) > 0) {
        vertex_indices.push_back(first_index.x);
        uv_indices.push_back(first_index.y);
        normal_indices.push_back(first_index.z);
        vertex_indices.push_back(last_index.x);
        uv_indices.push_back(last_index.y);
        normal_indices.push_back(last_index.z);
        vertex_indices.push_back(index.x);
        uv_indices.push_back(index.y);
        normal_indices.push_back(index.z);
        last_index = index;
      }
    }
  }
  fclose(fp);

  for (int i = 0; i < vertex_indices.size(); i++) {
    face_vertices.push_back(vertices[vertex_indices[i] - 1]);
  }
  for (int i = 0; i < uv_indices.size(); i++) {
    face_uvs.push_back(uvs[uv_indices[i] - 1]);
  }
  for (int i = 0; i < normal_indices.size(); i++) {
    face_normals.push_back(normals[normal_indices[i] - 1]);
  }
}


struct ReadObjMesh : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto mesh = zen::IObject::make<MeshObject>();
    readobj(path.c_str(), mesh->vertices, mesh->uvs, mesh->normals);
    set_output("mesh", mesh);
  }
};

static int defReadObjMesh = zen::defNodeClass<ReadObjMesh>("ReadObjMesh",
    { /* inputs: */ {
    }, /* outputs: */ {
    "mesh",
    }, /* params: */ {
    {"string", "path", ""},
    }, /* category: */ {
    "trimesh",
    }});

}
