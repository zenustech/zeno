#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <Light.hpp>
#include <Probe.hpp>
#include <vector>
#include <memory>

namespace zenvis
{
    struct Scene
    {
        Scene(const Scene &) = delete;
        Scene(Scene &&) = delete;
        Scene &operator=(const Scene &) = delete;
        Scene &operator=(Scene &) = delete;
        static Scene &getInstance()
        {
            static Scene scene;
            return scene;
        }

    private:
        Scene() = default;
        ~Scene() = default;

    public:
        static constexpr std::size_t maxLightsNum = 16;
        std::vector<std::unique_ptr<Light>> lights;
    
        Light *addLight()
        {
            if (lights.size() >= maxLightsNum)
            {
                return nullptr;
            }
            auto light = std::make_unique<Light>();
            auto pLight = light.get();
            lights.push_back(std::move(light));
            return pLight;
        }

        bool removeLight(std::size_t index)
        {
            if (index < 0 || index >= lights.size())
            {
                return false;
            }
            lights.erase(lights.begin() + index);
            return true;
        }

        static constexpr std::size_t maxProbesNum = 16;
        std::vector<std::unique_ptr<Probe>> probes;
    
        Probe *addProbe()
        {
            if (probes.size() >= maxProbesNum)
            {
                return nullptr;
            }
            auto probe = std::make_unique<Probe>();
            auto pProbe = probe.get();
            probes.push_back(std::move(probe));
            return pProbe;
        }

        bool removeProbe(std::size_t index)
        {
            if (index < 0 || index >= probes.size())
            {
                return false;
            }
            probes.erase(probes.begin() + index);
            return true;
        }

    }; // struct Scene

}; // namespace zenvis

#endif // #include __SCENE_HPP__