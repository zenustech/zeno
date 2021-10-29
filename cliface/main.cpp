#include <zeno/dop/dop.h>
#include <zeno/ztd/zany.h>
#include <zeno/types/Mesh.h>
#include <zeno/zycl/zycl.h>

USING_ZENO_NAMESPACE



int main()
{
#if 1
    zycl::vector<int> buf;

    {
        auto vec = buf.as_vector();
        printf("%zd\n", vec.size());
        for (int i = 0; i < 32; i++) {
            vec.push_back(i + 1);
        }
    }

    {
        auto hbuf = buf.get_access<zycl::access::mode::read>(zycl::host_handler{});
        for (int i = 0; i < 32; i++) {
            printf("%d\n", hbuf[i]);
        }
    }

#else
    auto n1 = dop::desc_of("ReadOBJMesh").create();
    n1->inputs.at(0) = dop::Input_Value{(std::string)"models/cube.obj"};
    n1->apply();
    auto mesh = zany_cast<std::shared_ptr<types::Mesh>>(n1->outputs.at(0));
    for (auto x: mesh->vert.to_vector()) {
        printf("%f %f %f\n", x[0], x[1], x[2]);
    }
#endif

    return 0;
}
