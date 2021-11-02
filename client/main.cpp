#if 0
#include <zeno/dop/dop.h>
#include <zeno/ztd/zany.h>
#include <zeno/types/Mesh.h>
#include <zeno/zycl/zycl.h>

USING_ZENO_NAMESPACE

int main()
{
    zycl::queue que;
#ifndef ZENO_SYCL_IS_EMULATED
    std::cout << "SYCL device: " << que.get_device().get_info<zycl::info::device::name>() << ", backend: " << que.get_backend() << std::endl;
#endif

#if 0
    zycl::vector<int> buf;

    {
        decltype(auto) vec = buf.as_vector();
        for (int i = 0; i < 32; i++) {
            vec.push_back(i + 1);
        }
    }

    buf.resize(40);

    {
        que.submit([&] (zycl::handler &cgh) {
            auto axr_buf = zycl::make_access<zycl::access::mode::discard_read_write>(cgh, buf);
            cgh.parallel_for(zycl::range<1>(buf.size()), [=] (zycl::item<1> idx) {
                axr_buf[idx] += 1;
            });
        });
    }

    buf.resize(48);

    {
        auto axr_buf = buf.get_access<zycl::access::mode::read>(zycl::host_handler{});
        for (int i = 0; i < buf.size(); i++) {
            printf("%d\n", axr_buf[i]);
        }
    }

#else

    auto n1 = dop::descriptor_table().at("ReadOBJMesh").create();
    n1->inputs.at(0) = dop::Input_Value{ztd::make_any<std::string>("models/cube.obj")};
    n1->apply();
    auto n2 = dop::descriptor_table().at("Transform").create();
    n2->inputs.at(0) = dop::Input_Value{n1->outputs.at(0)};
    n2->inputs.at(1) = dop::Input_Value{ztd::make_any<math::vec3f>({0.3f, 0.5f, 0.1f})};
    n2->inputs.at(2) = dop::Input_Value{ztd::make_any<math::vec3f>({1, 1, 1})};
    n2->inputs.at(3) = dop::Input_Value{ztd::make_any<math::vec4f>({0, 0, 0, 1})};
    n2->apply();
    auto mesh = pointer_cast<types::Mesh>(n2->outputs.at(0));

    for (auto x: mesh->vert.to_vector()) {
        printf("%f %f %f\n", x[0], x[1], x[2]);
    }

#endif

    return 0;
}
#else

#include <zeno/ztd/any_ptr.h>
#include <iostream>

USING_ZENO_NAMESPACE

template <class T1, class T2>
concept myconcept = requires (T1 t1, T2 t2) {
    math::clamp(t1, t2, 4);
};

int main()
{
    std::cout << myconcept<math::vec3i, math::vec4f> << std::endl;
    std::cout << myconcept<math::vec3i, math::vec3f> << std::endl;
    return 0;
}

#endif
