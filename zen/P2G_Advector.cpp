#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include <FastFLIP/FLIP_vdb.h>
//static void Advect(float dt, openvdb::points::PointDataGrid::Ptr m_particles, openvdb::Vec3fGrid::Ptr m_velocity,
//				   openvdb::Vec3fGrid::Ptr m_velocity_after_p2g, float pic_component, int RK_ORDER);

// I am thinking about create an empty result, only to indicate the order of nodes process?
// here this node is modifying two data simultaneously, particles, and velocity grid. which are all global data pointers
// I want this node to return an empty result(such as int 0), only to provide a connection to a next node. 
namespace zenbase{
    struct P2G_Advector : zen::INode{
        virtual void apply() override {
            auto dt = std::get<float>(get_param("time_step"));
            auto smoothness = std::get<float>(get_param("pic_smoothness"));
            auto RK_ORDER = std::get<int>(get_param("RK_ORDER"));
            auto particles = get_input("Particles")->as<VDBPointsGrid>();
            auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
            auto velocity_after_p2g = get_input("PostAdvVelocity")->as<VDBFloat3Grid>();
            FLIP_vdb::Advect(dt, particles->m_grid, velocity->m_grid, velocity_after_p2g->m_grid, smoothness, RK_ORDER);

            // oh isee, you're tring to return a 'int', which is not an zen::IObject now
            // we currently only have Matrix, Mesh, Particles as IObject
            // primitive types like int, can only be parameters (by get_param) now
            // should I add them? I 
            //yes please, ok, we shall be able to return a defalut Int, only serve as a way to provide a node connection.
            //so, this 'int' is not really necessary 'int', but just a dummy socket? we don't need 'float' and 'vec3' by this mean?right
            //oh then we'd better add DummyObject instead of IntObject (does the int value matter here?) doesn't at all! I see
            //then, by default, all nodes shall also accept dummyObject as input to indicate its predicessor (by predicessor you mean the order of exec)right
            //so let's add an input called "ConnectDummy" like this (see below)
        }
    };
}

static int defMeshToSDF = zen::defNodeClass<P2G_Advector>("P2G_Advector",
    { /* inputs: */ {
        "Particles", "Velocity", "PostAdvVelocity",
        // add this to nodes that want to accept dummy (want this automatically added to all nodes?)yes, although this is optional, but just let every node have this feature
    }, /* outputs: */ {
        // and, I guess, you proberly want to add dummy output to all nodes too?not sure, do we support multiply outputs now?
    // yes, at least in the editor UI, Ok then add this
    }, /* params: */ {
    {"float", "time_step", "0.04"},
    {"int", "RK_ORDER", "233333"},
    {"float", "pic_smoothness", "233.3"},
    }, /* category: */ {
    "openvdb",
    }});

}


//by the way, did you and dilei prefer 4 tabs than 2 tabs? I happened to have adapt it in taichi's code style in my vimrc..
//i think you're leaving this computer...
