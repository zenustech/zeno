#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/utils/string.h>

namespace zeno {

zany parseHeatmapObj(const std::string& json)
{
    auto heatmap = std::make_shared<zeno::HeatmapObject>();
    rapidjson::Document doc;
    doc.Parse(json.c_str());

    if (!doc.IsObject() || !doc.HasMember("nres") || !doc.HasMember("color"))
        return nullptr;
    int nres = doc["nres"].GetInt();
    std::string ramps = doc["color"].GetString();
    std::stringstream ss(ramps);
    std::vector<std::pair<float, zeno::vec3f>> colors;
    int count;
    ss >> count;
    for (int i = 0; i < count; i++) {
        float f = 0.f, x = 0.f, y = 0.f, z = 0.f;
        ss >> f >> x >> y >> z;
        //printf("%f %f %f %f\n", f, x, y, z);
        colors.emplace_back(
            f, zeno::vec3f(x, y, z));
    }

    for (int i = 0; i < nres; i++) {
        float fac = i * (1.f / nres);
        zeno::vec3f clr;
        for (int j = 0; j < colors.size(); j++) {
            auto [f, rgb] = colors[j];
            if (f >= fac) {
                if (j != 0) {
                    auto [last_f, last_rgb] = colors[j - 1];
                    auto intfac = (fac - last_f) / (f - last_f);
                    //printf("%f %f %f %f\n", fac, last_f, f, intfac);
                    clr = zeno::mix(last_rgb, rgb, intfac);
                }
                else {
                    clr = rgb;
                }
                break;
            }
        }
        heatmap->colors.push_back(clr);
    }
    return heatmap;
}

}
