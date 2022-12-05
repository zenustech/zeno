#include "zenovis/LiveManager.h"
#include <zenovis/Camera.h>
#include "httplib.h"
#include "json.hpp"
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/matrix_decompose.hpp>

using namespace httplib;

namespace zenovis {
    LiveManager::LiveManager(){
        std::cout << "LiveManager Constructed\n";
        verLoadCount = 0;
        camLoadCount = 0;

        std::thread([&]() {
            try
            {
                std::cout << "LiveManager: Server Running.\n";
                Server svr;
                svr.Get("/hi", [](const Request& req, Response& res) {
                    res.set_content("Hello World!", "text/plain");
                });
                svr.Post("/ver", [&](const auto& req, auto& res) {
                    std::cout << "ver req.body " << req.body << "\n";
                    verLoadCount++;
                    ParseHttpJson4Mesh(req.body);
                    res.set_content("Ver Received", "text/plain");
                });
                svr.Post("/cam", [&](const auto& req, auto& res) {
                    std::cout << "cam req.body " << req.body << "\n";
                    camLoadCount++;
                    ParseHttpJson4Cam(req.body);
                    res.set_content("Cam Received", "text/plain");
                });
                svr.Get("/stop", [&](const Request& req, Response& res) {
                    svr.stop();
                });
                svr.listen("localhost", 5236);

            }
            catch(std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }
        }).detach();
        std::cout << "LiveManager Listening\n";
    }

    LiveManager::~LiveManager(){

    }

    void LiveManager::ParseHttpJson4Cam(std::string data){
        using json = nlohmann::json;
        json parseData = json::parse(data);
        int translation_size = parseData["translation"].size();

        CameraIngredient ingredient;
        ingredient.translation = parseData["translation"].get<CAMERA_TRANS>();

        std::cout << " translation_size " << translation_size << "\n";
        GenerateCameraObject(ingredient);
    }

    void LiveManager::ParseHttpJson4Mesh(std::string data) {
        using json = nlohmann::json;
        json parseData = json::parse(data);
        int vertices_size = parseData["vertices"].size();
        int vertexCount_size = parseData["vertexCount"].size();
        int vertexList_size = parseData["vertexList"].size();

        PrimIngredient ingredient;
        ingredient.vertices = parseData["vertices"].get<VERTICES>();
        ingredient.vertexCount = parseData["vertexCount"].get<VERTEX_COUNT>();
        ingredient.vertexList = parseData["vertexList"].get<VERTEX_LIST>();

        std::cout << " vertices_size " << vertices_size << " vertexCount_size " << vertexCount_size
                  << " vertexList_size " << vertexList_size << "\n";

        GeneratePrimitiveObject(ingredient);
    }
    void LiveManager::GeneratePrimitiveObject(PrimIngredient& ingredient) {
        primObject = std::make_shared<zeno::PrimitiveObject>();
        auto& vert = primObject->verts;
        auto& loops = primObject->loops;
        auto& polys = primObject->polys;
        for(int i=0; i<ingredient.vertices.size(); i++){
            auto& v = ingredient.vertices[i];
            vert.emplace_back(v[0], v[1], v[2]);
        }
        int start = 0;
        for(int i=0; i<ingredient.vertexCount.size(); i++){
            auto count = ingredient.vertexCount[i];
            for(int j=start; j<start+count; j++){
                loops.emplace_back(ingredient.vertexList[j]);
            }
            polys.emplace_back(i * count, count);

            start += count;
        }
    }

    void LiveManager::GenerateCameraObject(CameraIngredient ingredient){
        cameraObject = std::make_shared<zeno::CameraObject>();
        float transX = ingredient.translation[0];
        float transY = ingredient.translation[1];
        float transZ = ingredient.translation[2];
        float rotateX = ingredient.translation[3];
        float rotateY = ingredient.translation[4];
        float rotateZ = ingredient.translation[5];
        float scaleX = ingredient.translation[6];
        float scaleY = ingredient.translation[7];
        float scaleZ = ingredient.translation[8];

        glm::mat4 transMatrixR = glm::translate(glm::vec3(transX, transY, -transZ));
        glm::mat4 transMatrixL = glm::translate(glm::vec3(transX, transY, transZ));
        float ax = rotateX * (M_PI / 180.0);
        float ay = rotateY * (M_PI / 180.0);
        float az = rotateZ * (M_PI / 180.0);
        glm::mat3 mx = glm::mat3(1,0,0,  0,cos(ax),-sin(ax),  0,sin(ax),cos(ax));
        glm::mat3 my = glm::mat3(cos(ay),0,sin(ay),  0,1,0,  -sin(ay),0,cos(ay));
        glm::mat3 mz = glm::mat3(cos(az),-sin(az),0,  sin(az),cos(az),0,  0,0,1);
        auto rotateMatrix3 = mx*my*mz;
        auto rotateMatrix4 = glm::mat4((rotateMatrix3));

        //auto matrix = transMatrixL * rotateMatrix4 * transMatrixR;
        auto matrix = rotateMatrix4;
        glm::vec3 trans, scale, skew; glm::quat rot; glm::vec4 perp;
        glm::decompose(matrix, trans, rot, scale, skew, perp);
        glm::mat3 rotMatrix = glm::mat3_cast(rot);

        zeno::CameraData cameraData;
        cameraData.pos = zeno::vec3f(transX, transY, transZ);
        cameraData.view = zeno::vec3f(rotMatrix[2][0], rotMatrix[2][1], rotMatrix[2][2]);
        cameraData.up = -zeno::vec3f(rotMatrix[1][0], rotMatrix[1][1], rotMatrix[1][2]);
        std::cout << "RotateMatrix\n\t" << rotMatrix[0][0] << " " << rotMatrix[0][1] << " " << rotMatrix[0][2]
                  << "\n\t" << rotMatrix[1][0] << " " << rotMatrix[1][1] << " " << rotMatrix[1][2]
                  << "\n\t" << rotMatrix[2][0] << " " << rotMatrix[2][1] << " " << rotMatrix[2][2] << "\n";
        std::cout << "pos " <<  trans[0] << " " << trans[1] << " " << trans[2] << "\n";
        std::cout << "view " <<  cameraData.view[0] << " " << cameraData.view[1] << " " << cameraData.view[2] << "\n";
        std::cout << "up " <<  cameraData.up[0] << " " << cameraData.up[1] << " " << cameraData.up[2] << "\n";
        scene->camera->setCamera(cameraData);
    }
}