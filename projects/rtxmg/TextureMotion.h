#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace zeno::cluster {
// Two-view finite differences predict a short primary-ray displacement.
// No scene-unit threshold: per-surface/tile bounds are applied on the GPU.
struct TextureView {
    std::array<float,3> eye{},right{1,0,0},up{0,1,0},front{0,0,1};
    float horizontal=1,vertical=1;
};
struct TexturePrediction {
    std::array<float,3> eye{},right{},up{},front{};
    float spread=1;
    bool active=false,reset=false;
};
class TextureMotion {
    TextureView previous{};
    bool valid=false;
public:
    void reset(){valid=false;}
    TexturePrediction update(const TextureView& view,float frames) {
        TexturePrediction out;
        auto finite=[](const auto& a){return std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);});};
        if(!finite(view.eye)||!finite(view.right)||!finite(view.up)||!finite(view.front)||
           !std::isfinite(view.horizontal)||!std::isfinite(view.vertical)||view.horizontal<=0||view.vertical<=0||
           !std::isfinite(frames)||frames<=0){valid=false;out.reset=true;return out;}
        auto dot=[](const auto& a,const auto& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
        if(valid) {
            const float x=previous.horizontal/view.horizontal,y=previous.vertical/view.vertical;
            out.reset=dot(view.front,previous.front)<.94f || dot(view.right,previous.right)<.94f ||
                x<.8f||x>1.25f||y<.8f||y>1.25f;
            if(!out.reset) {
                frames=std::clamp(frames,1.f,4.f);
                float change=0;
                for(int i=0;i<3;++i) {
                    out.eye[i]=frames*(view.eye[i]-previous.eye[i]);
                    out.right[i]=frames*(view.right[i]-x*previous.right[i]);
                    out.up[i]=frames*(view.up[i]-y*previous.up[i]);
                    out.front[i]=frames*(view.front[i]-previous.front[i]);
                    change+=std::fabs(out.eye[i])+std::fabs(out.right[i])+std::fabs(out.up[i])+std::fabs(out.front[i]);
                }
                out.spread=std::clamp(1.f+frames*(1.f-y),.5f,2.f);
                out.active=change>1.e-6f;
            }
        }
        previous=view;valid=true;return out;
    }
};
}
