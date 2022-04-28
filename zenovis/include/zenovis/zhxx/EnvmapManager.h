#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stb_image.h>
#include <unordered_map>
#include <zeno/utils/log.h>
#include <zenovis/zhxx/EnvmapIntegrator.h>
#include <zenovis/opengl/common.h>

namespace zenovis {

struct EnvmapManager {
    /* begin zhxx happy */
    GLuint envTexture = (GLuint)-1;
    std::unordered_map<std::string, GLuint> envTextureCache;
    std::unique_ptr<EnvmapIntegrator> integrator;

    Scene *scene;

    explicit EnvmapManager(Scene *scene)
        : scene(scene), integrator(std::make_unique<EnvmapIntegrator>(scene)) {
    }

    unsigned int loadCubemap(std::vector<std::string> faces) {
        unsigned int textureID(-1);
        CHECK_GL(glGenTextures(1, &textureID));
        CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, textureID));

        int width, height, nrChannels;
        for (unsigned int i = 0; i < faces.size(); i++) {
            unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
            if (data) {
                CHECK_GL(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB,
                             width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data));
                stbi_image_free(data);
            } else {
                zeno::log_warn("Cubemap tex failed to load at path: {}", faces[i]);
                CHECK_GL(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB,
                             1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr));
            }
        }
        CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));

        return textureID;
    }

    unsigned int setup_env_map(std::string name) {
        if (envTextureCache.count(name)) {
            envTexture = envTextureCache.at(name);
        } else {
            std::vector<std::string> faces{
                "assets/sky_box/" + name + "/right.jpg",
                "assets/sky_box/" + name + "/left.jpg",
                "assets/sky_box/" + name + "/top.jpg",
                "assets/sky_box/" + name + "/bottom.jpg",
                "assets/sky_box/" + name + "/front.jpg",
                "assets/sky_box/" + name + "/back.jpg",
            };
            envTexture = loadCubemap(faces);
            envTextureCache[name] = envTexture;
        }
        integrator->preIntegrate(envTexture);
        return envTexture;
    }

    void ensureGlobalMapExist() {
        if (envTexture == (GLuint)-1)
            setup_env_map("Default");
    }

    unsigned int getGlobalEnvMap() {
        return envTexture;
    }
};

} // namespace zenovis
