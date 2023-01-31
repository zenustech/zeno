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

using std::string;
using std::unordered_set;
using std::unordered_map;
using std::function;
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

class Picker {
  public:
    static Picker& GetInstance() {
        static Picker instance;
        return instance;
    }
    void pick(int x, int y);
    void pick(int x0, int y0, int x1, int y1);
    void add(const string& prim_name);
    string just_pick_prim(int x, int y);
    const unordered_set<string>& get_picked_prims();
    const unordered_map<string, unordered_set<int>>& get_picked_elems();
    void sync_to_scene();
    void load_from_str(const string& str, int mode);
    string save_to_str(int mode);
    void save_context();
    void load_context();
    void focus(const string& prim_name);
    void clear();
    void set_picked_elems_callback(function<void(unordered_map<string, unordered_set<int>>&)>);

  private:
    Picker() {
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        picker = zenovis::makeFrameBufferPicker(scene);
        select_mode_context = -1;
    };

    std::unique_ptr<zenovis::IPicker> picker;

    function<void(unordered_map<string, unordered_set<int>>&)> picked_elems_callback;

    unordered_set<string> selected_prims;
    unordered_map<string, unordered_set<int>> selected_elements;

    int select_mode_context;
    unordered_set<string> selected_prims_context;
    unordered_map<string, unordered_set<int>> selected_elements_context;

    string focused_prim;
};

}

#endif //VIEWPORT_PICKER_H
