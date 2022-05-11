#include "assimp/scene.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"

#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/logger.h>

#include <stack>


void readNode(std::vector<zeno::vec3f> &vertices,
              std::vector<zeno::vec3i> &indices,
              aiNode* node,
              aiScene const* scene){
    zeno::log_info("----- Node Name {}", node->mName.C_Str());
    unsigned int m_indicesIncrease = 0;

    for(unsigned int i=0; i<node->mNumMeshes; i++){
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        for(unsigned int j = 0; j < mesh->mNumVertices; j++){
            zeno::vec3f vec(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertices.push_back(vec);
        }

        for(unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            zeno::vec3i incs(face.mIndices[0]+m_indicesIncrease,
                             face.mIndices[1]+m_indicesIncrease,
                             face.mIndices[2]+m_indicesIncrease);
            indices.push_back(incs);
        }

        m_indicesIncrease += mesh->mNumVertices;
    }
}

void readFBXFile(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3i> &indices,
        const char *fbx_path
        )
{
    Assimp::Importer importer;
    aiScene const* scene = importer.ReadFile(fbx_path,
                                             aiProcess_Triangulate
                                             | aiProcess_FlipUVs
                                             | aiProcess_CalcTangentSpace
                                             | aiProcess_JoinIdenticalVertices);
    if(! scene)
        zeno::log_error("ReadFBXPrim: Invalid assimp scene");

    zeno::log_info("----- Num Animation {}", scene->mNumAnimations);

    std::stack<aiNode*> stack;
    stack.push(scene->mRootNode);

    while(true){
        if(stack.empty()){
            break;
        }

        aiNode* node = stack.top();
        stack.pop();

        if(node->mNumChildren > 0){
            for(unsigned int i=0; i<node->mNumChildren; i++){
                readNode(vertices, indices, node->mChildren[i], scene);
                stack.push(node->mChildren[i]);
            }
        }else{
            readNode(vertices, indices, node, scene);
        }
    }

    zeno::log_info("ReadFBXPrim: readFBXFile done.");
}



struct ReadFBXPrim : zeno::INode {

    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        auto &tris = prim->tris;

        zeno::log_info("ReadFBXPrim: path {}\n", path);

        readFBXFile(pos, tris, path.c_str());

        zeno::log_info("ReadFBXPrim: vertices {}", pos.size());
        zeno::log_info("ReadFBXPrim: indices {}", tris.size());

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadFBXPrim,
        {       /* inputs: */
            {
                {"readpath", "path"},
            },  /* outputs: */
            {
                {"prim"},
            },  /* params: */
            {

            },  /* category: */
            {
                "primitive",
           }
       });
