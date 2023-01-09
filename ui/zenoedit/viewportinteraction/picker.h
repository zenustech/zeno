#ifndef VIEWPORT_PICKER_H
#define VIEWPORT_PICKER_H
#include <zeno/utils/vec.h>
#include <viewport/zenovis.h>

#include <QtWidgets>

#include <cfloat>
#include <optional>
#include <unordered_set>
// TODO need a more elegant way to solve callbacks

#define CMP(x, y) \
	(fabsf(x - y) <= FLT_EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

using std::string;
using std::unordered_set;
using std::unordered_map;
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

struct PickingContext{
    int select_mode;
    std::unordered_set<std::string> selected_objects;
    std::unordered_map<std::string, std::unordered_set<int>> selected_elements;

    void saveContext() {
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        select_mode = scene->select_mode;
        selected_objects = std::move(scene->selected);
        selected_elements = std::move(scene->selected_elements);
    }

    void loadContext() {
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        if (select_mode < 0 || select_mode > 3) scene->select_mode = 0;
        else scene->select_mode = select_mode;
        scene->selected = std::move(selected_objects);
        scene->selected_elements = std::move(selected_elements);
    }
};

class Picker {
  public:
    static Picker& GetInstance() {
        static Picker instance;
        return instance;
    }
//    void pickWithRay(QVector3D ray_ori, QVector3D ray_dir,
//                     const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete);
//    void pickWithRay(QVector3D cam_pos, QVector3D left_up, QVector3D left_down, QVector3D right_up, QVector3D right_down,
//                     const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete);
    void pick(int x, int y);
    void pick(int x0, int y0, int x1, int y1);
    string just_pick_prim(int x, int y);
    const unordered_set<string>& get_picked_prims();
    const unordered_map<string, unordered_set<int>>& get_picked_elems();
    void sync_to_scene();
    void setTarget(const string& prim_name);
    void bindNode(const QModelIndex& node, const QModelIndex& subgraph, const std::string& sock_name);
    void unbindNode();
  private:
    Picker() : need_sync(false) {
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        picker = zenovis::makeFrameBufferPicker(scene);
    };

    std::unique_ptr<zenovis::IPicker> picker;
    std::vector<string> prim_set;
    bool need_sync;
    QModelIndex node;
    QModelIndex subgraph;
    string sock_name;
    inline void onPrimitiveSelected();
    void syncResultToNode();

    unordered_set<string> selected_prims;
    unordered_map<string, unordered_set<int>> selected_elements;
};

}

#endif //VIEWPORT_PICKER_H
