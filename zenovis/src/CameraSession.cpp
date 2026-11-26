#include <zenovis/CameraSession.h>
#include <zeno/utils/FrameCapture.h>
#include <tinygltf/json.hpp>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace zenovis::camera_session {
static_assert(Count<=96,"Camera session exceeds numeric recorder capacity");
namespace {
struct State {
    std::mutex mutex;
    std::atomic<bool> running{false};
    std::atomic<uint64_t> epoch{0},marks{0};
    std::unique_ptr<zeno::framecapture::Recorder> writer;
    Status latest;
    uint32_t scenes{};
};
State& state(){static State value;return value;}
struct Pending {Frame frame;uint64_t epoch{};bool valid=false;};
thread_local Pending pending;
thread_local Frame replayFrame;
thread_local bool replaying=false;
void writeJson(const std::filesystem::path& file,const nlohmann::json& data){
    std::ofstream output(file);if(!output)throw std::runtime_error("Cannot write camera recording: "+file.u8string());
    output<<data.dump(2);output.flush();if(!output)throw std::runtime_error("Cannot finish camera recording: "+file.u8string());
}
}
const std::vector<std::string>& fields(){
    static const std::vector<std::string> names={
#define ZENO_FIELD_NAME(name) #name,
        ZENO_CAMERA_SESSION_FIELDS(ZENO_FIELD_NAME)
#undef ZENO_FIELD_NAME
    };return names;
}
bool active(){return state().running.load(std::memory_order_acquire);}
uint64_t generation(){return state().epoch.load(std::memory_order_acquire);}
bool start(const std::string& directory){
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    if(s.running)return false;
    try {
        const auto root=std::filesystem::u8path(directory);
        std::filesystem::create_directories(root);
        for(const auto& p:std::filesystem::directory_iterator(root))
            if(p.path().filename().u8string().find("camera-")==0)throw std::runtime_error("Choose a new capture directory");
        writeJson(root/"schema.json",{{"version",1},{"fields",fields()},{"mode","CPU camera/timing capture; no extra CUDA queries"}});
        std::string columns;for(const auto& name:fields()){if(!columns.empty())columns+=',';columns+=name;}
        s.writer=std::make_unique<zeno::framecapture::Recorder>(root,"camera",columns.c_str(),true);
        const auto epoch=++s.epoch;s.latest={true,0,0,epoch,directory,{}};s.scenes=0;s.marks=0;
        s.running.store(true,std::memory_order_release);return true;
    }catch(const std::exception& e){s.latest.error=e.what();return false;}
}
Status stop(){
    auto& s=state();std::unique_ptr<zeno::framecapture::Recorder> writer;Status result;
    {std::lock_guard<std::mutex> lock(s.mutex);s.running=false;
        if(s.writer)s.latest.droppedBatches=s.writer->droppedBatches();
        s.latest.markers=s.marks.load();
        s.latest.active=false;result=s.latest;writer=std::move(s.writer);}
    // Drain/close outside the control mutex. Rendering never waits for disk IO.
    if(writer){writer->finish();result.droppedBatches=writer->droppedBatches();if(!writer->error().empty())result.error=writer->error();}writer.reset();
    if(!result.directory.empty())try {
        writeJson(std::filesystem::u8path(result.directory)/"status.json",{{"frames",result.frames},{"markers",result.markers},{"dropped_batches",result.droppedBatches},{"complete",result.error.empty()&&!result.droppedBatches},{"error",result.error}});
    }catch(const std::exception& e){result.error=e.what();}
    {std::lock_guard<std::mutex> lock(s.mutex);if(s.latest.generation==result.generation)s.latest=result;}
    return result;
}
Status status(){auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);auto result=s.latest;
    if(s.running)result.markers=s.marks.load();
    if(s.writer){result.droppedBatches=s.writer->droppedBatches();if(!s.writer->error().empty())result.error=s.writer->error();}return result;}
void mark(){auto& s=state();if(s.running)++s.marks;}
uint32_t scene(const std::string& metadata){
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);if(!s.writer)return 0;
    const auto id=++s.scenes;
    try {writeJson(std::filesystem::u8path(s.latest.directory)/("scene-"+std::to_string(id)+".json"),nlohmann::json::parse(metadata));}
    catch(const std::exception& e){s.latest.error=e.what();}
    return id;
}
void publishRender(const Frame& frame){if(!active())return;pending={frame,generation(),true};}
void finishViewport(double startMs,double durationMs,int w,int h,int ratio){
    if(!active()||!pending.valid)return;
    auto& s=state();auto frame=pending.frame;const auto epoch=pending.epoch;pending.valid=false;
    frame[window_width]=w;frame[window_height]=h;frame[render_ratio]=ratio;frame[marker]=double(s.marks.load());
    std::lock_guard<std::mutex> lock(s.mutex);if(!s.writer||epoch!=s.latest.generation)return;
    s.writer->record(startMs,durationMs,frame.values.data(),frame.values.size());++s.latest.frames;
}
void validate(const Frame& f){
    for(double x:f.values)if(!std::isfinite(x))throw std::runtime_error("Camera replay contains a non-finite value");
    auto integer=[&](Field k,double lo,double hi){const auto x=f[k];if(x<lo||x>hi||x!=std::floor(x))throw std::runtime_error("Invalid camera replay field: "+fields()[k]);};
    integer(width,16,16384);integer(height,16,16384);integer(samples,1,10000);integer(max_bounce,1,255);
    integer(window_width,16,65536);integer(window_height,16,65536);integer(render_ratio,1,64);
    integer(seed,0,4294967295.);integer(sample_start,0,4294967295.);integer(history_start,0,4294967295.);
    for(auto k:{simple_render,denoise,sharc,adaptive_lod,keep_history,two_sided,moving,clear_textures,aces,exposure,panorama,vr180,freeze_textures,freeze_geometry,show_background,sharc_debug,sharc_demodulation,sharc_sh,need_aov})integer(k,0,1);
    integer(sharc_spacing,0,65536);integer(sharc_accumulation,0,65536);integer(sharc_stale,0,65536);
    integer(picking,0,2);integer(debug_view,0,7);integer(history_limit,4,256);
    if(f[focal_length]<=0||f[sensor_width]<=0||f[sensor_height]<=0||f[aperture]<0||f[focal_distance]<f[focal_length])throw std::runtime_error("Invalid camera replay lens");
    for(auto first:{right_x,up_x,front_x}) {const auto n=f.values[first]*f.values[first]+f.values[first+1]*f.values[first+1]+f.values[first+2]*f.values[first+2];if(n<.9||n>1.1)throw std::runtime_error("Invalid camera replay basis");}
}
void setReplay(const Frame& frame){validate(frame);replayFrame=frame;replaying=true;}
bool replay(Frame& frame){if(!replaying)return false;frame=replayFrame;return true;}
void clearReplay(){replaying=false;}
}
