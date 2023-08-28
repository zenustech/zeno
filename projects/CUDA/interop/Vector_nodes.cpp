#include "Vector.hpp"
#include <fmt/core.h>
#include <tuple>
#include <variant>
#include <zeno/zeno.h>

namespace zeno
{
    struct MakeZsVectorObject : INode
    {
        void apply() override
        {
            // TODO
            auto input_size = get_input2<int>("size");
            auto input_memsrc = get_input2<std::string>("memsrc");
            auto intput_devid = get_input2<int>("dev_id");
            // auto input_virtual = get_input2<bool>("virtual");
            auto intput_elem_type = get_input2<std::string>("elem_type");

            zs::memsrc_e memsrc;
            if (input_memsrc == "host")
                memsrc = zs::memsrc_e::host;
            else if (input_memsrc == "device")
                memsrc = zs::memsrc_e::device;
            else
                memsrc = zs::memsrc_e::um;

#define MAKE_VECTOR_OBJ_T(T)                                                                   \
    if (intput_elem_type == #T)                                                                \
    {                                                                                          \
        auto allocator = zs::get_memory_source(memsrc, static_cast<zs::ProcID>(intput_devid)); \
        vectorObj->set(zs::Vector<T, zs::ZSPmrAllocator<false>>{allocator, 0});                \
    }

            auto vectorObj = std::make_shared<VectorViewLiteObject>();
            MAKE_VECTOR_OBJ_T(int)
            MAKE_VECTOR_OBJ_T(float)
            MAKE_VECTOR_OBJ_T(double)
            std::visit([input_size](auto &vec)
                       { vec.resize(input_size); },
                       vectorObj->value);

            set_output("ZsVector", std::move(vectorObj));
        }
    };

    //  memsrc, size, elem_type, dev_id, virtual
    ZENDEFNODE(MakeZsVectorObject, {
                                     {{"int", "size", "0"},
                                      {"enum host device um", "memsrc", "device"},
                                      {"int", "dev_id", "0"},
                                      //   {"bool", "virtual", "false"},
                                      {"enum float int double", "elem_type", "float"}},
                                     {"ZsVector"},
                                     {},
                                     {"PyZFX"},
                                 });
}