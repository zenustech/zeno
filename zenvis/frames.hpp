#pragma once

#include <cstring>
#include <vector>
#include <memory>
#include <string>
#include <map>


namespace zenvis {


struct ObjectData {
  std::unique_ptr<std::vector<char>> memory;
  std::unique_ptr<std::vector<char>> shader;
  std::string type;
};


struct FrameData {
  std::vector<std::unique_ptr<ObjectData>> objects;
};


extern std::map<int, std::unique_ptr<FrameData>> frames;


}
