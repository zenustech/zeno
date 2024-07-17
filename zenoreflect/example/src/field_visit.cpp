#include "test.h"

using namespace zeno::reflect;

int main() {

    // 推荐in-place构造一个Any，然后拿指针出来用
    Any instance = make_any<zeno::Hhhh>();
    // 拿指针
    zeno::Hhhh* typed_instance_ptr = any_cast<zeno::Hhhh>(&instance);
    // 当然也可以拿引用
    zeno::Hhhh& typed_instance_ref = any_cast<zeno::Hhhh&>(instance);

    // === 上面三个都是指向同一个对象，在多线程环境下要注意数据竞争问题 ===

    // 我们使用基类指针
    IReflectedObject* obj = typed_instance_ptr;

    // 通过虚函数获取真正的类型信息
    // 记得判空, 这个里面可能为空, 用了空的TypeHandle会导致运行时错误
    TypeHandle type_info = obj->type_info();
    
    // 获取字段操作类，它仅储存了操作，并不储存数据。数据等到操作发生时才传入。
    for (auto field : type_info->get_member_fields()) {
        // 找到我们要的
        std::cout << field->get_name().c_str() << std::endl;

        // 判断它的名字
        if (field->get_name() == "a1") {
            // 作为普通类型, 我们需要知道它的类型才能正确转换指针
            if (field->get_field_type() == get_type<int>()) {
                // 获取字段数据指针
                int* a1_ptr = field->get_field_ptr_typed<int>(instance);
                std::cout << "a1的值: " << *a1_ptr << std::endl;

                // 修改一下它的值
                *a1_ptr = 123;
            }
        }
    }

    // 输出一下修改后的instance
    std::cout << instance << std::endl;

    return 0;
}
