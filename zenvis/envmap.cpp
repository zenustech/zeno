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
static unsigned int loadCubemap(std::vector<std::string> faces)
{
    unsigned int textureID = 0xffffffffu;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++)
    {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                         0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
            );
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap tex failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}  
void setup_env_map(std::string name)
{
  if (envTextureCache.count(name)) {
    envTexture = envTextureCache.at(name);
  } else {
    std::vector<std::string> faces
    {
      fmt::format("assets/sky_box/{}/right.jpg", name),
      fmt::format("assets/sky_box/{}/left.jpg", name),
      fmt::format("assets/sky_box/{}/top.jpg", name),
      fmt::format("assets/sky_box/{}/bottom.jpg", name),
      fmt::format("assets/sky_box/{}/front.jpg", name),
      fmt::format("assets/sky_box/{}/back.jpg", name)
    };
    envTexture = loadCubemap(faces);
    envTextureCache[name] = envTexture;
  }
}
unsigned int getGlobalEnvMap() {
    return envTexture;
}
/* end zhxx happy */
}