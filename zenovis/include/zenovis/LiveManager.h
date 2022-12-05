#pragma once

#include <zenovis/Scene.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/core/IObject.h>
#include <string>
#include <memory>

namespace zenovis {

    struct LiveManager : zeno::disable_copy {

        typedef std::vector<std::vector<float>> VERTICES;
        typedef std::vector<int> VERTEX_COUNT;
        typedef std::vector<int> VERTEX_LIST;
        typedef std::vector<float> CAMERA_TRANS;

        struct PrimIngredient{
            VERTICES vertices;
            VERTEX_COUNT vertexCount;
            VERTEX_LIST vertexList;
        };

        struct CameraIngredient{
            CAMERA_TRANS translation;
        };

        LiveManager();
        ~LiveManager();

        void ParseHttpJson4Mesh(std::string data);
        void ParseHttpJson4Cam(std::string data);
        void GeneratePrimitiveObject(PrimIngredient& ingredient);
        void GenerateCameraObject(CameraIngredient ingredient);

        std::shared_ptr<zeno::PrimitiveObject> primObject;
        std::shared_ptr<zeno::CameraObject> cameraObject;

        Scene* scene;
        int verLoadCount;
        int camLoadCount;
    };

}
