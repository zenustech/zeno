#include "MyShader.hpp"
#include "stdafx.hpp"
#include "main.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <array>
#include <stb_image_write.h>
#include <Hg/OpenGL/stdafx.hpp>
#include <unordered_map>
#include "zenvisapi.hpp"
#include <spdlog/spdlog.h>

namespace zenvis {
/* begin zhxx happy */
static GLuint envTexture = (GLuint)-1;
static std::unordered_map<std::string, GLuint> envTextureCache;
static unsigned int loadCubemap(std::string path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    // set the texture wrapping/filtering options (on the currently bound texture object)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load and generate the texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    return textureID;
}  
void setup_env_map(std::string name) {
  if (envTextureCache.count(name)) {
    envTexture = envTextureCache.at(name);
  } else {
    std::string path = fmt::format("assets/sky_sphere/{}.jpg", name);
    fmt::print("{}\n", path);
    envTexture = loadCubemap(path);
    envTextureCache[name] = envTexture;
  }
}
unsigned int getGlobalEnvMap() {
    return envTexture;
}
/* end zhxx happy */
}
