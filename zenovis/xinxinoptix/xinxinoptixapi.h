#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cctype>

#include <glm/glm.hpp>
#include "optixSphere.h"
#include "zeno/utils/vec.h"
#include "zeno/types/CurveType.h"
#include "zeno/types/LightObject.h"

#include "Portal.h"
#include "optiXStuff.h"

enum ShaderMark {
    Mesh, Sphere, Volume,

    CURVE_QUADRATIC,
    CURVE_RIBBON,
    CURVE_CUBIC,

    CURVE_LINEAR,
    CURVE_BEZIER,
    CURVE_CATROM,
};

typedef std::tuple<std::string, ShaderMark> shader_key_t;
struct ByShaderKey
{
    bool operator()(const shader_key_t& a, const shader_key_t& b) const
    {
        auto& [a1, a2] = a;
        auto& [b1, b2] = b;
    
        if (a2 < b2)
            return true;
        if (a2 > b2)
            return false;

        // if (a1=="Default")
        //     return true;
        if (b1 == "Default")
            return false;
        if (a1 == "Light")
            return false;
        if (a1 < b1 || b1=="Light")
            return true;
        else
            return false;
    
        return a1 < b1;
    }

    size_t operator()(const shader_key_t& key) const
    {
        return hash(key);
    }

    static size_t hash(const shader_key_t& key) noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(std::get<0>(key));
        std::size_t h2 = std::hash<int>{}(std::get<1>(key));
        return h1 ^ (h2 << 1);
    }

    static bool equal( const shader_key_t& x, const shader_key_t& y ) {
        return std::get<0>(x) == std::get<0>(y) && std::get<1>(x) == std::get<1>(y);
    }

};

static const std::map<zeno::CurveType, ShaderMark> CURVE_SHADER_MARK {
    { zeno::CurveType::QUADRATIC_BSPLINE, ShaderMark::CURVE_QUADRATIC },
    { zeno::CurveType::RIBBON_BSPLINE,    ShaderMark::CURVE_RIBBON    },
    { zeno::CurveType::CUBIC_BSPLINE,     ShaderMark::CURVE_CUBIC     },
    
    { zeno::CurveType::LINEAR,    ShaderMark::CURVE_LINEAR    },
    { zeno::CurveType::BEZIER,    ShaderMark::CURVE_BEZIER    },
    { zeno::CurveType::CATROM,    ShaderMark::CURVE_CATROM    },
};

struct ShaderPrepared {
    bool dirty = true;

    ShaderMark mark;
    std::string matid;
    std::string filename;

    std::string callable;
    std::string parameters;

    std::map<std::string, std::string> macros;
    
    std::vector<std::shared_ptr<OptixUtil::cuTexture>> texs;
    std::vector<std::string>       vdb_keys;
};

struct VDBGridKeyChannels {
    std::string requested;
    std::string resolved;
    bool valid = false;
};

inline VDBGridKeyChannels parseVDBGridKeyChannels(const std::string& vdb_key) {
    VDBGridKeyChannels channels;

    const auto close = vdb_key.rfind("}#");
    if (close == std::string::npos) {
        return channels;
    }

    const auto open = vdb_key.rfind('{', close);
    if (open == std::string::npos || open + 1 >= close) {
        return channels;
    }

    const auto split = vdb_key.find('|', open + 1);
    if (split == std::string::npos || split >= close) {
        return channels;
    }

    channels.requested = vdb_key.substr(open + 1, split - open - 1);
    channels.resolved = vdb_key.substr(split + 1, close - split - 1);
    channels.valid = true;
    return channels;
}

inline std::string normalizeVDBChannelName(const std::string& channel) {
    std::string normalized;
    normalized.reserve(channel.size());
    for (unsigned char ch : channel) {
        if (ch == '_' || ch == '-' || ch == ' ') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

inline bool isDensityVDBChannelName(const std::string& channel) {
    const auto normalized = normalizeVDBChannelName(channel);
    return normalized == "density" || normalized == "dens" || normalized == "rho";
}

inline bool vdbKeyLooksLikeDensity(const std::string& vdb_key) {
    const auto channels = parseVDBGridKeyChannels(vdb_key);
    if (!channels.valid) {
        return isDensityVDBChannelName(vdb_key);
    }
    return isDensityVDBChannelName(channels.requested) || isDensityVDBChannelName(channels.resolved);
}

inline size_t selectDensityVDBSlot(const std::vector<std::string>& vdb_keys, size_t max_slots = 8) {
    size_t fallback = vdb_keys.size();
    const auto count = std::min(vdb_keys.size(), max_slots);
    for (size_t i = 0; i < count; ++i) {
        if (vdb_keys[i].empty()) {
            continue;
        }
        if (fallback == vdb_keys.size()) {
            fallback = i;
        }
        if (vdbKeyLooksLikeDensity(vdb_keys[i])) {
            return i;
        }
    }
    return fallback == vdb_keys.size() ? 0 : fallback;
}

struct DensityVDBSlotRefs {
    uint8_t primary_slot = 0;
    std::set<uint8_t> referenced_slots {};
};

inline void appendVDBSlotsReferencedByDensityCode(
    const std::string& density_code,
    std::vector<size_t>& ordered_slots,
    size_t max_slots = 8)
{
    const std::string token = "vdb_grids[";
    size_t pos = 0;
    while ((pos = density_code.find(token, pos)) != std::string::npos) {
        pos += token.size();
        size_t slot = 0;
        bool has_digit = false;
        while (pos < density_code.size()) {
            const char c = density_code[pos];
            if (c < '0' || c > '9') {
                break;
            }
            has_digit = true;
            slot = slot * 10 + static_cast<size_t>(c - '0');
            ++pos;
        }
        if (!has_digit || pos >= density_code.size() || density_code[pos] != ']' || slot >= max_slots) {
            continue;
        }
        if (std::find(ordered_slots.begin(), ordered_slots.end(), slot) == ordered_slots.end()) {
            ordered_slots.push_back(slot);
        }
    }
}

inline DensityVDBSlotRefs resolveDensityVDBSlots(
    const std::vector<std::string>& vdb_keys,
    const std::string& density_code,
    size_t max_slots = 8)
{
    DensityVDBSlotRefs refs {};
    const size_t count = std::min(vdb_keys.size(), max_slots);
    std::vector<size_t> ordered_slots;
    appendVDBSlotsReferencedByDensityCode(density_code, ordered_slots, max_slots);

    for (const size_t slot : ordered_slots) {
        if (slot >= count || vdb_keys[slot].empty()) {
            continue;
        }
        if (refs.referenced_slots.empty()) {
            refs.primary_slot = slot;
        }
        refs.referenced_slots.insert(slot);
    }

    if (refs.referenced_slots.empty()) {
        refs.primary_slot = selectDensityVDBSlot(vdb_keys, max_slots);
        if (refs.primary_slot < count && !vdb_keys[refs.primary_slot].empty()) {
            refs.referenced_slots.insert(refs.primary_slot);
        }
    }

    return refs;
}

namespace xinxinoptix {


void optixCleanup();
void optixDestroy();

void optixrender(int fbo = 0, int samples = 1, bool denoise = false, bool simpleRender = false);
void *optixgetimg(int &w, int &h);
void optixinit(int argc, char* argv[]);
void optixupdatebegin();

void prepareScene();
void updateShaders(std::vector<std::shared_ptr<ShaderPrepared>> &shaders, 
                    bool requireTriangObj, bool requireTriangLight, 
                    bool requireSphereObj, bool requireSphereLight, 
                    bool requireVolumeObj, uint usesCurveTypes, bool refresh=false);
                    
void updateRootIAS();
void buildLightTree();
void configPipeline(bool dirty);
void configPipeline(bool dirty, bool pipelineDirty);

void set_window_size(int nx, int ny);
void set_outside_random_number(unsigned int outside_random_number);
void set_perspective(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, float fpd, float aperture);
void set_physical_camera_param(float aperture, float shutter_speed, float iso, int scale, bool aces, bool exposure, bool panorama_camera, bool panorama_vr180, float pupillary_distance);
void set_perspective_by_fov(float const *U, float const *V, float const *W, float const *E, float aspect, float fov, int fov_type, float L, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift);
void set_perspective_by_focal_length(float const *U, float const *V, float const *W, float const *E, float aspect, float focal_length, float w, float h, float focal_distance, float aperture, float pitch, float yaw, float h_shift, float v_shift);

void get_click_pos(float x, float y, std::function<void(glm::vec3)> cbClickPosSig);
void get_click_id(float x, float y, std::function<void(std::tuple<std::string, std::string, uint32_t>)> cbClickIdSig);

struct LightDat {
    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> v2;
    std::vector<float> normal;
    std::vector<float> color;

    float spreadMajor;
    float spreadMinor;
    float intensity;
    float vIntensity;
    float fluxFixed;
    float maxDistance;
    float falloffExponent;

    bool visible, doubleside;
    uint8_t shape, type;
    uint16_t mask;

    uint32_t coordsBufferOffset = UINT_MAX;
    uint32_t normalBufferOffset = UINT_MAX;

    std::string profileKey;
    std::string textureKey;
    float textureGamma;
};

void load_triangle_light(std::string const &key, LightDat &ld,
                        const zeno::vec3f &v0,  const zeno::vec3f &v1,  const zeno::vec3f &v2, 
                        const zeno::vec3f *pn0, const zeno::vec3f *pn1, const zeno::vec3f *pn2,
                        const zeno::vec3f *uv0, const zeno::vec3f *uv1, const zeno::vec3f *uv2);
                        
void load_light(std::string const &key, LightDat &ld, float const*v0, float const*v1, float const*v2);
                
void unload_light();
void update_procedural_sky(zeno::vec2f sunLightDir, float sunLightSoftness, zeno::vec2f windDir, float timeStart, float timeSpeed,
                           float sunLightIntensity, float colorTemperatureMix, float colorTemperature);
void update_hdr_sky(float sky_rot, zeno::vec3f sky_rot3d, float sky_strength);
void update_hdr_sky(zeno::vec3f sky_rot3d, float sky_strength);
glm::vec3 realtime_rotate_sky(glm::vec3 angle_vec);
void using_hdr_sky(bool enable);
void show_background(bool enable);

void updatePortalLights(const std::vector<Portal>& portals);
void updateDistantLights(std::vector<zeno::DistantLightData>& dldl);
// void optixUpdateUniforms(std::vector<float4> & inConstants);
void optixUpdateUniforms(void *inConstants, std::size_t size);

const std::map<std::string, LightDat> &get_lightdats();

}
