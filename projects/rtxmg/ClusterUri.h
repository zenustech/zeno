#pragma once
#include <filesystem>
#include <stdexcept>
#include <string>

namespace zeno::cluster {
// glTF URIs use percent-encoded UTF-8, independently of the host code page.
inline std::filesystem::path assetUriPath(const std::filesystem::path& root,const std::string& uri) {
    auto hex=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;return -1;};
    std::string path;path.reserve(uri.size());
    if(uri.find("://")!=std::string::npos || uri.rfind("data:",0)==0)
        throw std::runtime_error("Expected a local external glTF asset URI: "+uri);
    for(size_t i=0;i<uri.size();++i) {
        char c=uri[i];
        if(c=='%') {
            if(i+2>=uri.size()||hex(uri[i+1])<0||hex(uri[i+2])<0)throw std::runtime_error("Invalid percent escape in asset URI: "+uri);
            c=char((hex(uri[i+1])<<4)|hex(uri[i+2]));i+=2;
        } else if(c=='?'||c=='#')throw std::runtime_error("Local asset URI query/fragment is unsupported: "+uri);
        if(!c)throw std::runtime_error("NUL in asset URI");
        path+=c;
    }
    return (root/std::filesystem::u8path(path)).lexically_normal();
}
}
