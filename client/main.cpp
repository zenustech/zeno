#if 1
#include <zeno/dop/dop.h>
#include <zeno/ztd/zany.h>
#include <zeno/types/Mesh.h>
#include <zeno/zycl/parallel.h>

USING_ZENO_NAMESPACE

int main()
{
#if 1
    zycl::vector<int> buf;

    {
        decltype(auto) vec = buf.as_vector();
        for (int i = 0; i < 32; i++) {
            vec.push_back(i + 1);
        }
    }

    buf.resize(40);

    zycl::default_queue().submit([&] (zycl::handler &cgh) {
        auto axr_buf = zycl::make_access<zycl::access::mode::discard_read_write>(cgh, buf);
        zycl::parallel_for
        ( cgh
        , zycl::range<1>(buf.size())
        , [=] (zycl::item<1> idx) {
            axr_buf[idx] += 1;
        });
    });

    buf.resize(48);

    {
        auto axr_buf = zycl::host_access<zycl::access::mode::read>(buf);
        for (int i = 0; i < buf.size(); i++) {
            printf("%d\n", axr_buf[i]);
        }
    }

    /*zycl::default_queue().submit([&] (zycl::handler &cgh) {
        auto axr_buf = zycl::make_access<zycl::access::mode::read>(cgh, buf);
        zycl::parallel_reduce
        ( cgh
        , zycl::range<1>(buf.size())
        , zycl::range<1>(8)
        , buf
        , 0.0f
        , [] (auto x, auto y) { return x + y; }
        , [=] (zycl::item<1> idx, auto &reducer) {
            reducer.combine(axr_buf[idx]);
        });
    });*/

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

    /*auto n3 = dop::descriptor_table().at("Reduction").create();
    n3->inputs.at(0) = dop::Input_Value{n2->outputs.at(0)};
    n3->inputs.at(1) = dop::Input_Value{ztd::make_any<std::string>("centroid")};
    n3->apply();*/

    auto mesh = pointer_cast<types::Mesh>(n2->outputs.at(0));
    for (auto x: mesh->vert.to_vector()) {
        printf("%f %f %f\n", x[0], x[1], x[2]);
    }

    /*auto res1 = value_cast<math::vec3f>(n3->outputs.at(0));
    auto res2 = value_cast<math::vec3f>(n3->outputs.at(1));
    printf("%f %f %f\n", res1[0], res1[1], res1[2]);
    printf("%f %f %f\n", res2[0], res2[1], res2[2]);*/

#endif

    return 0;
}
#else

#include <zeno/ztd/any_ptr.h>
#include <iostream>

USING_ZENO_NAMESPACE

template <class T>
concept mama = requires (T t) {
    math::sqrt(t);
};

int main()
{
    std::cout << mama<float> << std::endl;
    return 0;
}

#endif
