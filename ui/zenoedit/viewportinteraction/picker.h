#ifndef VIEWPORT_PICKER_H
#define VIEWPORT_PICKER_H
#include <zeno/utils/vec.h>
#include <viewport/zenovis.h>

#include <QtWidgets>

#include <cfloat>
#include <optional>
#include <unordered_set>

#define CMP(x, y) \
	(fabsf(x - y) <= FLT_EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

class ViewportWidget;

namespace zeno {

std::optional<float> ray_box_intersect(
    zeno::vec3f const &bmin,
    zeno::vec3f const &bmax,
    zeno::vec3f const &ray_pos,
    zeno::vec3f const &ray_dir
);

bool test_in_selected_bounding(
    QVector3D centerWS,
    QVector3D cam_posWS,
    QVector3D left_normWS,
    QVector3D right_normWS,
    QVector3D up_normWS,
    QVector3D down_normWS
);

class Picker
{
public:
    Picker(ViewportWidget* pViewport);
    void initialize();
    void pick(int x, int y);
    void pick(int x0, int y0, int x1, int y1);
    void pick_depth(int x, int y);
    void add(const std::string& prim_name);
    std::string just_pick_prim(int x, int y);
    const std::unordered_set<std::string>& get_picked_prims();
    const std::unordered_map<std::string, std::unordered_set<int>>& get_picked_elems();
    void sync_to_scene();
    void load_from_str(const std::string& str, int mode);
    std::string save_to_str(int mode);
    void save_context();
    void load_context();
    void focus(const std::string& prim_name);
    void clear();
    void set_picked_depth_callback(std::function<void(float, int, int)>);
    void set_picked_elems_callback(std::function<void(std::unordered_map<std::string, std::unordered_set<int>>&)>);
    bool is_draw_mode();
    void switch_draw_mode();

private:
    zenovis::Scene* scene() const;

    std::unique_ptr<zenovis::IPicker> picker;
    
    ViewportWidget* m_pViewport;

    std::function<void(float, int, int)> picked_depth_callback;
    std::function<void(std::unordered_map<std::string, std::unordered_set<int>>&)> picked_elems_callback;

    std::unordered_set<std::string> selected_prims;
    std::unordered_map<std::string, std::unordered_set<int>> selected_elements;

    int select_mode_context;
    std::unordered_set<std::string> selected_prims_context;
    std::unordered_map<std::string, std::unordered_set<int>> selected_elements_context;

    std::string focused_prim;
    bool draw_mode;
};

}

#endif //VIEWPORT_PICKER_H
