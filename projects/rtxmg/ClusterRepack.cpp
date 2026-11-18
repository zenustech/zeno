#include "ClusterCache.h"
#include <fstream>
#include <iostream>
#include <cstring>
namespace fs=std::filesystem;
using namespace zeno::cluster;
int main(int argc,char** argv)try{
 if(argc!=2)throw std::runtime_error("Usage: zeno_cluster_repack cache-directory (baker must be stopped)");
 const auto root=fs::absolute(argv[1]);uint64_t before=0,after=0;size_t meshes=0;
 for(const auto& file:fs::directory_iterator(root)){
  if(file.path().extension()!=".json"||file.path().filename().string().find("mesh-")!=0)continue;
  auto meta=readJson(file.path());if(meta.value("pageCodec",0)==4)continue;
  const auto pages=root/meta.at("pages").get<std::string>();const auto temporary=pages.string()+".repack";
  std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
  const auto oldSize=fs::file_size(pages);
  Geometry geometry;geometry.pages=pages;
  for(const auto& g:meta.at("groups")){
   Group group;group.offset=g.at("offset");group.bytes=g.at("bytes");group.vertices=g.at("vertices");group.triangles=g.at("triangles");
   group.refined=g.at("refined").get<std::vector<int32_t>>();
   if(group.offset>oldSize||group.bytes>oldSize-group.offset)throw std::runtime_error("Invalid page file range while repacking");
   geometry.groups.push_back(std::move(group));
  }
  uint32_t id=0;
  for(auto& group:meta.at("groups")){
   const auto page=readPage(geometry,id++);
   group["offset"]=uint64_t(output.tellp());group["bytes"]=writePage(output,page);
  }
  output.close();if(!output)throw std::runtime_error("Page repack close failed");
  meta["pageBytes"]=fs::file_size(temporary);meta["pageCodec"]=4;
  writeJson(file.path().string()+".repack",meta);
  // Recoverable transaction: retain originals until both replacements exist.
  const auto backup=pages.string()+".before-repack",metaBackup=file.path().string()+".before-repack";
  fs::rename(pages,backup);fs::rename(file.path(),metaBackup);
  try{fs::rename(temporary,pages);fs::rename(file.path().string()+".repack",file.path());}
  catch(...){if(fs::exists(pages))fs::remove(pages);fs::rename(backup,pages);if(fs::exists(file.path()))fs::remove(file.path());fs::rename(metaBackup,file.path());throw;}
  fs::remove(backup);fs::remove(metaBackup);before+=oldSize;after+=meta["pageBytes"].get<uint64_t>();++meshes;
  if(meshes%25==0)std::cout<<"[cluster repack] meshes="<<meshes<<" beforeBytes="<<before<<" afterBytes="<<after<<std::endl;
 }
 std::cout<<"[cluster repack] meshes="<<meshes<<" beforeBytes="<<before<<" afterBytes="<<after<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"[cluster repack] "<<e.what()<<std::endl;return 1;}
