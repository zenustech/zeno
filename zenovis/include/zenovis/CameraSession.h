#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zenovis::camera_session {
// Numeric fields shared by the recorder and replay. The CSV/schema carries
// their names so changing the layout cannot silently corrupt an old capture.
#define ZENO_CAMERA_SESSION_FIELDS(X) \
 X(render_start_ms) X(renderer_ms) X(geometry_ms) X(launch_ms) X(request_ms) X(readback_ms) X(setup_ms) X(geometry_wait_ms) \
 X(width) X(height) X(window_width) X(window_height) X(render_ratio) \
 X(eye_x) X(eye_y) X(eye_z) X(right_x) X(right_y) X(right_z) X(up_x) X(up_y) X(up_z) X(front_x) X(front_y) X(front_z) \
 X(horizontal_shift) X(vertical_shift) X(pitch) X(yaw) X(focal_length) X(aperture) X(focal_distance) X(sensor_width) X(sensor_height) \
 X(samples) X(simple_render) X(denoise) X(max_bounce) X(sharc) X(debug_view) X(adaptive_lod) X(lod_pixel_error) X(keep_history) X(history_limit) X(two_sided) \
 X(seed) X(sample_start) X(history_start) X(moving) X(picking) X(click_x) X(click_y) X(clear_textures) X(geometry_version) X(effective_lod) X(geometry_pending) \
 X(physical_aperture) X(shutter_speed) X(iso) X(aces) X(exposure) X(panorama) X(vr180) X(pupil_distance) \
 X(freeze_textures) X(freeze_geometry) X(marker) X(scene_id) X(show_background) \
 X(sharc_spacing) X(sharc_scale) X(sharc_accumulation) X(sharc_stale) X(sharc_roughness) X(sharc_debug) X(sharc_demodulation) X(sharc_sh) X(need_aov) \
 X(texture_ticket_slots) X(texture_batches) X(texture_cpu_tasks) X(texture_uploads) X(texture_prepared) X(texture_feedback_contexts) \
 X(texture_callbacks_queued) X(texture_callbacks_active) X(texture_oldest_batch_ms) X(texture_oldest_callback_ms)
enum Field {
#define ZENO_FIELD_ENUM(name) name,
    ZENO_CAMERA_SESSION_FIELDS(ZENO_FIELD_ENUM)
#undef ZENO_FIELD_ENUM
    Count
};
struct Frame {
    std::array<double,Count> values{};
    double& operator[](Field field){return values[field];}
    double operator[](Field field) const{return values[field];}
};
struct Status {bool active{};uint64_t frames{},droppedBatches{},generation{};std::string directory,error;uint64_t markers{};};
const std::vector<std::string>& fields();
bool start(const std::string& directory);
Status stop();
Status status();
bool active();
uint64_t generation();
void mark();
// Called only when the renderer's scene/configuration identity changes.
uint32_t scene(const std::string& metadataJson);
void publishRender(const Frame& frame);
void finishViewport(double startMs,double durationMs,int windowWidth,int windowHeight,int ratio);
void setReplay(const Frame& frame);
bool replay(Frame& frame);
void clearReplay();
void validate(const Frame& frame);
}
