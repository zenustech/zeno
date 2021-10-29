#include <zeno/dop/dop.h>
#include <zeno/ztd/zany.h>
#include <zeno/types/Mesh.h>
#include <zeno/zycl/zycl.h>

USING_ZENO_NAMESPACE

int main()
{
    zycl::queue que;

#if 1
    zycl::vector<int> buf;

    {
        decltype(auto) vec = buf.as_vector();
        for (int i = 0; i < 32; i++) {
            vec.push_back(i + 1);
        }
    }

    {
        que.submit([&] (zycl::handler &cgh) {
            auto axr_buf = zycl::make_access<zycl::access::mode::discard_read_write>(cgh, buf);
            cgh.parallel_for(zycl::range<1>(buf.size()), [=] (zycl::item<1> idx) {
                axr_buf[idx] += 1;
            });
        });
    }

    {
        auto axr_buf = buf.get_access<zycl::access::mode::read>(zycl::host_handler{});
        for (int i = 0; i < buf.size(); i++) {
            printf("%d\n", axr_buf[i]);
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
