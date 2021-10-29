#include <zeno/dop/dop.h>
#include <zeno/ztd/zany.h>
#include <zeno/types/Mesh.h>
#include <zeno/zycl/zycl.h>

USING_ZENO_NAMESPACE



int main()
{
#ifndef ZENO_SYCL_IS_EMULATED
    decltype(auto) dev = zycl::queue().get_device();
    std::cout << "Zeno SYCL running on: " << dev.get_info<zycl::info::device::name>() << std::endl;
#endif

#if 1
    zycl::vector<int> buf(32);

    /*{
        auto vec = buf.as_vector();
        for (int i = 0; i < 32; i++) {
            vec.push_back(i + 1);
        }
    }*/

    {
        auto hbuf = buf.get_access<zycl::access::mode::read>(zycl::host_handler{});
        for (int i = 0; i < buf.size(); i++) {
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
