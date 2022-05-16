#include <zenovis/RenderEngine.h>

namespace zenovis {

std::map<std::string, std::function<std::unique_ptr<RenderEngine>(Scene *)>> RenderManager::factories;

}
