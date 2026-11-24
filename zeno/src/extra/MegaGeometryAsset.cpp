#include <zeno/extra/MegaGeometryAsset.h>
#include <zeno/utils/log.h>
#include "../../../projects/rtxmg/ClusterUri.h"
#include <tinygltf/json.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace zeno {
namespace {
namespace fs=std::filesystem;
using Json=nlohmann::json;
Json read(const fs::path& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot open Mega Geometry input: "+path.u8string());
    Json value;f>>value;return value;
}
bool baked(const Json& j) {return j.value("schema","")=="zeno.rtxmg.asset" && j.value("version",0)==1 && j.value("gray",false);}
bool inside(const fs::path& path,const fs::path& directory) {
    auto p=path.begin(),d=directory.begin();
    for(;d!=directory.end();++d,++p) {
        if(p==path.end())return false;
#ifdef _WIN32
        if(_wcsicmp(p->c_str(),d->c_str()))return false;
#else
        if(*p!=*d)return false;
#endif
    }
    return true;
}
std::string key(const Json& stamp) {
    uint64_t hash=14695981039346656037ull;
    for(unsigned char c:stamp.dump()){hash^=c;hash*=1099511628211ull;}
    std::ostringstream s;s<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return s.str();
}
#ifdef _WIN32
struct Handle {
    HANDLE h=INVALID_HANDLE_VALUE;
    explicit Handle(HANDLE value=INVALID_HANDLE_VALUE):h(value){}
    ~Handle(){if(h && h!=INVALID_HANDLE_VALUE)CloseHandle(h);}
    Handle(const Handle&)=delete;
};
struct Lock {
    Handle file;
    explicit Lock(const fs::path& path) {
        for(;;) {
            file.h=CreateFileW(path.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
            if(file.h!=INVALID_HANDLE_VALUE)break;
            if(GetLastError()!=ERROR_SHARING_VIOLATION)throw std::runtime_error("Cannot lock Mega Geometry cache: "+path.u8string());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};
std::wstring quote(const std::wstring& value) {
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:value) {
        if(c==L'\\'){++slashes;continue;}
        out.append(c==L'"'?slashes*2+1:slashes,L'\\');out+=c;slashes=0;
    }
    out.append(slashes*2,L'\\');return out+L'"';
}
void run(const fs::path& executable,const fs::path& source,const fs::path& output,unsigned jobs,const fs::path& log) {
    const std::vector<std::wstring> args{executable.wstring(),L"--source",source.wstring(),L"--output",output.wstring(),
        L"--include-instances",L"1",L"--attributes",L"1",L"--preserve-layers",L"1",L"--jobs",std::to_wstring(jobs)};
    std::wstring command;for(const auto& arg:args){if(!command.empty())command+=L' ';command+=quote(arg);}
    SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
    Handle logFile(CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&inherit,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr));
    Handle input(CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&inherit,OPEN_EXISTING,0,nullptr));
    Handle job(CreateJobObjectW(nullptr,nullptr));
    if(logFile.h==INVALID_HANDLE_VALUE || input.h==INVALID_HANDLE_VALUE || !job.h)
        throw std::runtime_error("Cannot initialize Mega Geometry baker process");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(!SetInformationJobObject(job.h,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))
        throw std::runtime_error("Cannot protect Mega Geometry baker process lifetime");
    STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES|STARTF_USESHOWWINDOW;
    startup.wShowWindow=SW_HIDE;startup.hStdInput=input.h;startup.hStdOutput=startup.hStdError=logFile.h;
    PROCESS_INFORMATION info{};
    if(!CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED,
        nullptr,nullptr,&startup,&info))throw std::runtime_error("Cannot start Mega Geometry baker: "+executable.u8string());
    Handle process(info.hProcess),thread(info.hThread);
    if(!AssignProcessToJobObject(job.h,process.h)) {
        TerminateProcess(process.h,1);WaitForSingleObject(process.h,INFINITE);
        throw std::runtime_error("Cannot attach Mega Geometry baker to its parent job");
    }
    ResumeThread(thread.h);WaitForSingleObject(process.h,INFINITE);
    DWORD code=1;GetExitCodeProcess(process.h,&code);
    if(code)throw std::runtime_error("Mega Geometry baking failed; see "+log.u8string());
}
#else
struct Lock {
    int fd=-1;
    explicit Lock(const fs::path& path){fd=open(path.c_str(),O_CREAT|O_RDWR,0666);if(fd<0||flock(fd,LOCK_EX))throw std::runtime_error("Cannot lock Mega Geometry cache");}
    ~Lock(){if(fd>=0)close(fd);}
};
void run(const fs::path& executable,const fs::path& source,const fs::path& output,unsigned jobs,const fs::path& log) {
    const auto pid=fork();if(pid<0)throw std::runtime_error("Cannot start Mega Geometry baker");
    if(!pid) {
        const int fd=open(log.c_str(),O_CREAT|O_WRONLY|O_TRUNC,0666);if(fd<0)_exit(126);
        dup2(fd,STDOUT_FILENO);dup2(fd,STDERR_FILENO);close(fd);
        const auto count=std::to_string(jobs);
        execl(executable.c_str(),executable.c_str(),"--source",source.c_str(),"--output",output.c_str(),
            "--include-instances","1","--attributes","1","--preserve-layers","1","--jobs",count.c_str(),nullptr);
        _exit(127);
    }
    int status=0;if(waitpid(pid,&status,0)!=pid || !WIFEXITED(status)||WEXITSTATUS(status))
        throw std::runtime_error("Mega Geometry baking failed; see "+log.u8string());
}
#endif
}
std::filesystem::path prepareMegaGeometryAsset(const fs::path& input,const fs::path& cacheDirectory,
    const fs::path& baker,unsigned jobs,unsigned revision) {
    auto source=fs::weakly_canonical(fs::absolute(input));
    if(fs::is_directory(source))source/="asset.json";
    const auto meta=read(source);
    if(baked(meta))return source;
    if(!meta.contains("asset") || !meta.at("asset").is_object() || meta.at("asset").value("version","")!="2.0" || !meta.contains("meshes"))
        throw std::runtime_error("MegaGeometryAsset expects a baked asset.json or raw glTF 2.0 source");
    if(cacheDirectory.empty())throw std::runtime_error("Set cacheDirectory outside the source asset folder to bake raw Mega Geometry");
    if(!jobs || jobs>64)throw std::runtime_error("Mega Geometry bakeJobs must be between 1 and 64");
    const auto cacheRoot=fs::weakly_canonical(fs::absolute(cacheDirectory));
    if(inside(cacheRoot,source.parent_path()))throw std::runtime_error("Mega Geometry cacheDirectory must be outside the source asset folder");
    std::set<fs::path> dependencies{source};
    for(const auto& buffer:meta.at("buffers"))
        dependencies.insert(fs::weakly_canonical(cluster::assetUriPath(source.parent_path(),buffer.at("uri").get<std::string>())));
    Json stamp{{"version",1},{"revision",revision},{"source",source.u8string()},
        {"attributes",true},{"instances",true},{"preserveLayers",true},{"files",Json::array()}};
    for(const auto& path:dependencies)stamp["files"].push_back({path.u8string(),fs::file_size(path),fs::last_write_time(path).time_since_epoch().count()});
    const auto cache=cacheRoot/key(stamp),manifest=cache/"asset.json",marker=cache/"source-stamp.json";
    fs::create_directories(cache);Lock lock(cache/"prepare.lock");
    if(fs::exists(marker) && fs::exists(manifest) && read(marker)==stamp && baked(read(manifest)))return manifest;
    if(!fs::is_regular_file(baker))throw std::runtime_error("Mega Geometry baker is missing: "+baker.u8string());
    log_info("Baking Mega Geometry {} into {}",source.u8string(),cache.u8string());
    run(baker,source,cache,jobs,cache/"bake.log");
    if(!baked(read(manifest)))throw std::runtime_error("Mega Geometry baker did not publish a valid asset manifest");
    const auto temp=cache/"source-stamp.json.tmp";
    {std::ofstream f(temp,std::ios::binary);f<<stamp.dump(2);if(!f)throw std::runtime_error("Cannot publish Mega Geometry cache identity");}
    if(fs::exists(marker))fs::remove(marker);fs::rename(temp,marker);
    return manifest;
}
} // namespace zeno
