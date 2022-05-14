#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include "assimp/scene.h"

#include "Definition.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

struct EvalAnim{
    double m_Duration;
    double m_TicksPerSecond;
    float m_CurrentFrame;
    float m_DeltaTime;

    NodeTree m_RootNode;

//    std::vector<Bone> m_Bones;
    std::unordered_map<std::string, aiMatrix4x4> m_Transforms;
//    std::unordered_map<std::string, BoneInfo> m_BoneInfoMap;
    std::unordered_map<std::string, BoneInfo> m_BoneOffset;
    std::unordered_map<std::string, Bone> m_Bones;
    std::vector<VertexInfo> m_Vertices;
    std::vector<unsigned int> m_Indices;

    void initAnim(std::shared_ptr<NodeTree>& nodeTree,
                  std::shared_ptr<BoneTree>& boneTree,
                  std::shared_ptr<FBXData>& fbxData,
                  std::shared_ptr<BoneOffset>& boneOffset){
        m_Duration = fbxData->duration;
        m_TicksPerSecond = fbxData->tick;
        m_Vertices = fbxData->vertices;
        m_Indices = fbxData->indices;

        m_RootNode = *nodeTree;
        m_Bones = boneTree->BoneMap;
//        m_BoneInfoMap = boneTree->BoneInfoMap;
        m_BoneOffset = boneOffset->BoneOffsetMap;

        m_CurrentFrame = 0.0f;

        // DEBUG
//        zeno::log_info("+++++ Bones Num {}", m_Bones.size());
//        zeno::log_info("+++++ Vert Num {}", m_Vertices.size());
//        zeno::log_info("+++++ Indices Num {}", m_Indices.size());
    }

    void updateAnimation(float dt, std::shared_ptr<zeno::PrimitiveObject>& prim) {
        m_DeltaTime = dt;
        m_CurrentFrame += m_TicksPerSecond * dt;
        m_CurrentFrame = fmod(m_CurrentFrame, m_Duration);

//        zeno::log_info("Update: Frame {}", m_CurrentFrame);

        calculateBoneTransform(&m_RootNode, aiMatrix4x4());
        calculateFinal(prim);
    }

    void calculateBoneTransform(const NodeTree *node, aiMatrix4x4 parentTransform) {
        std::string nodeName = node->name;
        aiMatrix4x4 nodeTransform = node->transformation;

        //zeno::log_info("Calculating: Node Name {}", nodeName);

//        Bone* bone = findBone(nodeName);

        if (m_Bones.find(nodeName) != m_Bones.end()) {
            auto& bone = m_Bones[nodeName];

            bone.update(m_CurrentFrame);
            nodeTransform = bone.m_LocalTransform;
        }
        aiMatrix4x4 globalTransformation = parentTransform * nodeTransform;

        if (m_BoneOffset.find(nodeName) != m_BoneOffset.end()) {  // found
            std::string boneName = m_BoneOffset[nodeName].name;
            aiMatrix4x4 boneOffset = m_BoneOffset[nodeName].offset;

            m_Transforms[boneName] = globalTransformation * boneOffset;
        }
        for (int i = 0; i < node->childrenCount; i++)
            calculateBoneTransform(&node->children[i], globalTransformation);
    }

    Bone *findBone(std::string const& name) {
//        auto iter = std::find_if(m_Bones.begin(), m_Bones.end(),
//                                 [&](const Bone& Bone)
//                                 {
//                                     return Bone.m_Name == name;
//                                 }
//        );
//        if (iter == m_Bones.end())
//            return nullptr;
//        else
//            return &(*iter);

//        if (m_Bones.find(name) == m_Bones.end())
//            return nullptr;
//        else
//            return &m_Bones[name];
    }

    void calculateFinal(std::shared_ptr<zeno::PrimitiveObject>& prim){
        auto &ver = prim->verts;
        auto &ind = prim->tris;

        for(unsigned int i=0; i<m_Vertices.size(); i++){
            auto& bwe = m_Vertices[i].boneWeights;
            auto& pos = m_Vertices[i].position;

            glm::vec4 tpos(0.0f, 0.0f, 0.0f, 0.0f);

            bool infd = false;

            for(auto& b: bwe){

                infd = true;
                auto& tr = m_Transforms[b.first];
                glm::mat4 trans = glm::mat4(tr.a1,tr.b1,tr.c1,tr.d1,
                                            tr.a2,tr.b2,tr.c2,tr.d2,
                                            tr.a3,tr.b3,tr.c3,tr.d3,
                                            tr.a4,tr.b4,tr.c4,tr.d4);
                glm::vec4 lpos = trans * glm::vec4(pos.x, pos.y, pos.z, 1.0f);
                tpos += lpos * b.second;
            }
            if(! infd)
                tpos = glm::vec4(pos.x, pos.y, pos.z, 1.0f);

            glm::vec3 fpos = glm::vec3(tpos.x/tpos.w, tpos.y/tpos.w, tpos.z/tpos.w);
            ver.emplace_back(fpos.x, fpos.y, fpos.z);
        }

        for(unsigned int i=0; i<m_Indices.size(); i+=3){
            zeno::vec3i incs(m_Indices[i],m_Indices[i+1],m_Indices[i+2]);
            ind.push_back(incs);
        }
    }
};

struct EvalFBXAnim : zeno::INode {

    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<zeno::NumericObject>("frameid")->get<int>();
        } else {
            frameid = zeno::state.frameid;
        }
//        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto fbxData = get_input<FBXData>("fbxdata");
        auto nodeTree = get_input<NodeTree>("nodetree");
        auto boneTree = get_input<BoneTree>("bonetree");
        auto boneOffset = get_input<BoneOffset>("boneoffset");

        EvalAnim anim;
        anim.initAnim(nodeTree, boneTree, fbxData, boneOffset);
        anim.updateAnimation(frameid/24.0f, prim);

        set_output("prim", std::move(prim));

//        zeno::log_info("EvalFBXAnim: Duration {} Tick {} Frame {}", fbxData->duration, fbxData->tick,
//                       frameid);
    }
};
ZENDEFNODE(EvalFBXAnim,
           {       /* inputs: */
               {
                   {"prim"}, {"frameid"},
                   {"FBXData", "fbxdata"},
                   {"NodeTree", "nodetree"},
                   {"BoneTree", "bonetree"},
                   {"BoneOffset", "boneoffset"}
               },  /* outputs: */
               {
                   "prim",
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });

