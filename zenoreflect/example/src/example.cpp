#include "data.h"
#include "reflect/container/object_proxy"
#include "reflect/container/any"
#include "reflect/container/arraylist"
#include <iostream>
#include <functional>

using namespace zeno::reflect;

namespace zeno {
    std::ostream& operator<<(std::ostream& os, const IAmPrimitve& p) {
        os << "i32 = " << p.i32;
        return os;
    }

}

void fix_console_encoding();

#ifdef _MSC_VER
#include <windows.h>

void fix_console_encoding() {
    SetConsoleOutputCP(CP_UTF8);
    std::locale::global(std::locale("en_US.UTF-8"));
}

#else
void fix_console_encoding() {}
#endif

int main(int argc, char* argv[]) {
    fix_console_encoding();

    // 获取类型名称
    TypeHandle handle = get_type<zeno::IAmPrimitve>();
    std::cout << "类型名称: " << handle->get_info().canonical_typename.c_str() << std::endl;

    zeno::IAmPrimitve hand_made_inst{};
    hand_made_inst.i32 = 789;
    std::cout << "直接构造: " << hand_made_inst << std::endl;

    TypeBase* type = handle.get_reflected_type_or_null();
    ITypeConstructor& ctor = type->get_constructor_checked({ zeno::reflect::type_info<int>(), zeno::reflect::type_info<std::string>() });

    std::cout << "这是调用构造函数所需的参数: ";
    for (const auto& t : ctor.get_params()) {
        std::cout << t.name() << "  ";
    }
    std::cout << std::endl;

    zeno::IAmPrimitve reflect_inst = ctor.create_instance_typed<zeno::IAmPrimitve>({ ctor.get_param_default_value(0), ctor.get_param_default_value(1) });
    std::cout << "使用反射调用拷贝构造函数创建的新实例: " << reflect_inst << std::endl;
    reflect_inst.i32 = 123;
    std::cout << "可以像正常对象一样访问: " << reflect_inst << std::endl;

    Any type_erased_inst = ctor.create_instance({ Any(678), ctor.get_param_default_value(1) });
    std::cout << "基于反射信息输出对象: \n" << type_erased_inst << std::endl;

    // 或者也可以通过直接赋值给Any 也支持移动语义
    Any use_operator_equal = hand_made_inst;
    std::cout << "直接赋值构造Any: \n" << use_operator_equal << std::endl;

    // 输出所有有DisplayName的反射类型
    {
        auto& registry = zeno::reflect::ReflectionRegistry::get();
        for (const auto& type : registry->all()) {
            if (const IRawMetadata * metadata = type->get_metadata()) {
                if (const IMetadataValue* value = metadata->get_value("DisplayName")) {
                    std::cout << "反射类型 " << type->get_info().canonical_typename.c_str() << " 的 DisplayName = \"" << value->as_string() << "\"" << std::endl;
                }
            }
        }
    }

    Any 我是一个Any = zeno::reflect::make_any<std::string>( "好吧" );
    std::cout << "类型是: " << 我是一个Any.type().name() << "\t值是: " << any_cast<std::string>(我是一个Any) << std::endl;

    zeno::reflect::Any simple_any = zeno::reflect::make_any<std::vector<std::string>>();
    if (simple_any.type() == zeno::reflect::type_info<std::vector<std::string>>()) {
        int a = 0;
    }

    // 测试赋值
    zeno::reflect::Any ann = zeno::reflect::make_any<int>(10);
    zeno::reflect::Any ann2 = ann;
    auto xx = ann2.has_value() ? zeno::reflect::any_cast<int>(ann2) : 0;


    return 0;
}
