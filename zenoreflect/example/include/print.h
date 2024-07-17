#pragma once

#include "reflect/type"
#include <iostream>
#include <string>

namespace zeno
{
namespace reflect
{
    std::ostream& operator<<(std::ostream& os, const Any& any) {
        // 检查any容器是否为空 也可以用any.has_value()
        if (any) {
            // 获取类型句柄 这玩意大小只有 aligned(sizeof(size_t) + sizeof(bool), sizeof(size_t)) 可以随意拷贝
            TypeHandle type_visitor = any.type();
            // 打印信息
            os << "===\t" << type_visitor->get_info().canonical_typename.c_str() << "\t===\n";
            // 获取类型所有字段的访问器
            const auto& fields =  type_visitor->get_member_fields();
            // 遍历类型字段
            for (const auto& field_visitor : fields) {
                // 打印字段名称
                os << "\t" << field_visitor->get_name().c_str() << "\t=\t";
                // 各种类型的输出适配器 这里只实现了int的
                if (field_visitor->get_field_type() == get_type<int>()) {
                    os << *field_visitor->get_field_ptr_typed<int>(any) << "\n";
                }
                if (field_visitor->get_field_type() == get_type<std::string>()) {
                    os << *field_visitor->get_field_ptr_typed<std::string>(any) << "\n";
                }
            }
            os << "===========================================";
        } else {
            os << "<nullany>";
        }
        return os;
    }
}
}

REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::string)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(char)
