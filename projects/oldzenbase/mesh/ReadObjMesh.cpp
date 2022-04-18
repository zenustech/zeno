#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/MeshObject.h>
#include <zeno/StringObject.h>
#include <cstring>
#include <omp.h>

namespace zeno {

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

  bool has_normal = false;
  bool has_texcoord = false;
  char hdr[128];
  while (EOF != fscanf(fp, "%s", hdr)) 
  {
    if (!strcmp(hdr, "v")) {
      glm::vec3 vertex;
      fscanf(fp, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
      vertices.push_back(vertex);

    } else if (!strcmp(hdr, "vt")) {
      has_texcoord = true;
      glm::vec2 uv;
      fscanf(fp, "%f %f\n", &uv.x, &uv.y);
      uvs.push_back(uv);

    } else if (!strcmp(hdr, "vn")) {
      has_normal = true;
      glm::vec3 normal;
      fscanf(fp, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
      normals.push_back(normal);

    } else if (!strcmp(hdr, "f")) {
      if(has_normal&&has_texcoord)
      {
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
      }else if(has_normal)
      {
        glm::uvec3 last_index, first_index, index;

        fscanf(fp, "%d/%d", &index.x, &index.y);
        first_index = index;

        fscanf(fp, "%d/%d", &index.x, &index.y);
        last_index = index;

        while (fscanf(fp, "%d/%d", &index.x, &index.y) > 0) {
          vertex_indices.push_back(first_index.x);
          normal_indices.push_back(first_index.y);
          vertex_indices.push_back(last_index.x);
          normal_indices.push_back(last_index.y);
          vertex_indices.push_back(index.x);
          normal_indices.push_back(index.y);
          last_index = index;
        }
      }else if(has_texcoord)
      {
        glm::uvec3 last_index, first_index, index;

        fscanf(fp, "%d/%d", &index.x, &index.y);
        first_index = index;

        fscanf(fp, "%d/%d", &index.x, &index.y);
        last_index = index;

        while (fscanf(fp, "%d/%d", &index.x, &index.y) > 0) {
          vertex_indices.push_back(first_index.x);
          uv_indices.push_back(first_index.y);
          vertex_indices.push_back(last_index.x);
          uv_indices.push_back(last_index.y);
          vertex_indices.push_back(index.x);
          uv_indices.push_back(index.y);
          last_index = index;
        }
      }else {
        //printf("face vert only\n");
        glm::uvec3 last_index, first_index, index;

        fscanf(fp, "%d", &index.x);
        first_index = index;

        fscanf(fp, "%d", &index.x);
        last_index = index;

        while (fscanf(fp, "%d", &index.x) > 0) {
          vertex_indices.push_back(first_index.x);
          vertex_indices.push_back(last_index.x);
          vertex_indices.push_back(index.x);
          last_index = index;
        }
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


struct ReadObjMesh : zeno::INode {
  virtual void apply() override {
    auto path = get_param<std::string>("path"));
    auto mesh = zeno::IObject::make<MeshObject>();
    readobj(path.c_str(), mesh->vertices, mesh->uvs, mesh->normals);
    set_output("mesh", mesh);
  }
};

static int defReadObjMesh = zeno::defNodeClass<ReadObjMesh>("ReadObjMesh",
    { /* inputs: */ {
    }, /* outputs: */ {
    "mesh",
    }, /* params: */ {
    {"readpath", "path", ""},
    }, /* category: */ {
    "trimesh",
    }});

struct MeshMix : zeno::INode {
  virtual void apply() override {
    auto meshA = get_input("meshA")->as<MeshObject>();
    auto meshB = get_input("meshB")->as<MeshObject>();
    auto coef = get_input("coef")->as<zeno::NumericObject>()->get<float>();
    auto mesh = zeno::IObject::make<MeshObject>();
    mesh->vertices=meshA->vertices;
    mesh->uvs=meshA->uvs;
    mesh->normals=meshA->normals;
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size();i++)
    {
      mesh->vertices[i] = (1.0f-coef)*meshA->vertices[i] + coef*meshB->vertices[i];
    }
    set_output("mesh", mesh);
  }
};

static int defMeshMix = zeno::defNodeClass<MeshMix>("MeshMix",
    { /* inputs: */ {
      "meshA",
      "meshB",
      "coef",
    }, /* outputs: */ {
    "mesh",
    }, /* params: */ {
    }, /* category: */ {
    "trimesh",
    }});
struct ImportObjMesh : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto mesh = zeno::IObject::make<MeshObject>();
    readobj(path->get().c_str(), mesh->vertices, mesh->uvs, mesh->normals);
    set_output("mesh", mesh);
  }
};

static int defImportObjMesh = zeno::defNodeClass<ImportObjMesh>("ImportObjMesh",
    { /* inputs: */ {
    "path",
    }, /* outputs: */ {
    "mesh",
    }, /* params: */ {
    }, /* category: */ {
    "trimesh",
    }});

}
