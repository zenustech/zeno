#include <zeno/zeno.h>
#include <zeno/types/IGeometryObject.h>


namespace zeno {

    using namespace zeno::reflect;

    struct CreateAttribute : INode {
        void apply() override {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            std::string attr_name = ZImpl(get_input2<std::string>("attr_name"));
            std::string m_type = ZImpl(get_input2<std::string>("Attribute Type"));
            std::string m_attrgroup = ZImpl(get_input2<std::string>("Attribute Group"));

            if (attr_name == "") {
                throw makeError<UnimplError>("the attribute name cannot be empty.");
            }

            AttrVar attr_value;
            if (m_type == "Vector2") {
                attr_value = ZImpl(get_input2<zeno::vec2f>("Vector2 Value"));
            }
            else if (m_type == "Vector3") {
                attr_value = ZImpl(get_input2<vec3f>("Vector3 Value"));
            }
            else if (m_type == "Vector4") {
                attr_value = ZImpl(get_input2<vec4f>("Vector4 Value"));
            }
            else if (m_type == "Integer") {
                attr_value = ZImpl(get_input2<int>("Integer Value"));
            }
            else if (m_type == "Float") {
                attr_value = ZImpl(get_input2<float>("Float Value"));
            }
            else if (m_type == "String") {
                attr_value = ZImpl(get_input2<std::string>("String Value"));
            }
            else if (m_type == "Boolean") {
                attr_value = ZImpl(get_input2<bool>("Boolean Value"));
            }

            if (auto spPrim = dynamic_cast<PrimitiveObject*>(input_object.get())) {
                throw makeError<UnimplError>("Unsupport Legacy Primitive Object for creating attribute");
            }
            else if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(input_object.get())) {
                if (m_attrgroup == "Point") {
                    spGeo->m_impl->create_attr(ATTR_POINT, attr_name, attr_value);
                }
                else if (m_attrgroup == "Face") {
                    spGeo->m_impl->create_attr(ATTR_FACE, attr_name, attr_value);
                }
                else if (m_attrgroup == "Geometry") {
                    spGeo->m_impl->create_attr(ATTR_GEO, attr_name, attr_value);
                }
            }
            else {
                throw makeError<UnimplError>("Unsupport Object for creating attribute");
            }
            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(CreateAttribute,
    {
        {
            {gParamType_Geometry, "Input"},
            ParamPrimitive("attr_name", gParamType_String, "attribute1"),
            ParamPrimitive("Attribute Type",  gParamType_String, "float", zeno::Combobox, std::vector<std::string>{"Vector2", "Vector3", "Vector4", "Integer", "Float", "String", "Boolean"}),
            ParamPrimitive("Attribute Group", gParamType_String, "Point", zeno::Combobox, std::vector<std::string>{"Point", "Face", "Geometry"}),
            ParamPrimitive("Integer Value", gParamType_Int, 0, zeno::Lineedit, Any(), "visible = parameter('Attribute Type').value == 'Integer';"),
            ParamPrimitive("Float Value", gParamType_Float, 0.f, zeno::Lineedit, Any(), "visible = parameter('Attribute Type').value == 'Float';"),
            ParamPrimitive("String Value", gParamType_String, "", zeno::Lineedit, Any(),"visible = parameter('Attribute Type').value == 'String';"),
            ParamPrimitive("Boolean Value", gParamType_Bool, false, zeno::Checkbox, Any(),"visible = parameter('Attribute Type').value == 'Boolean';"),
            ParamPrimitive("Vector2 Value", gParamType_Vec2f, vec2f(0,0), zeno::Vec2edit, Any(),"visible = parameter('Attribute Type').value == 'Vector2';"),
            ParamPrimitive("Vector3 Value", gParamType_Vec3f, vec3f(0,0,0), zeno::Vec3edit, Any(),"visible = parameter('Attribute Type').value == 'Vector3';"),
            ParamPrimitive("Vector4 Value", gParamType_Vec4f, vec4f(0,0,0,0), zeno::Vec4edit, Any(),"visible = parameter('Attribute Type').value == 'Vector4';")
        },
        {
            {gParamType_Geometry, "Output"}
        },
        {},
        {"geom"}
    });


    struct DeleteAttribute : INode {
        void apply() override {
            auto spGeo = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            std::string attr_name = ZImpl(get_input2<std::string>("attr_name"));
            std::string m_attrgroup = ZImpl(get_input2<std::string>("Attribute Group"));
            if (attr_name == "") {
                throw makeError<UnimplError>("the attribute name cannot be empty.");
            }

            auto attrnames = zeno::split_str(attr_name, ' ', false);
            for (auto attrname : attrnames) {
                if (m_attrgroup == "Point") {
                    spGeo->m_impl->delete_attr(ATTR_POINT, attrname);
                }
                else if (m_attrgroup == "Face") {
                    spGeo->m_impl->delete_attr(ATTR_FACE, attrname);
                }
                else if (m_attrgroup == "Geometry") {
                    spGeo->m_impl->delete_attr(ATTR_GEO, attrname);
                }
            }
            ZImpl(set_output("Output", std::move(spGeo)));
        }
    };
    ZENDEFNODE(DeleteAttribute,
    {
        {
            {gParamType_Geometry, "Input"},
            ParamPrimitive("attr_name", gParamType_String, "attribute1"),
            ParamPrimitive("Attribute Group", gParamType_String, "Point", zeno::Combobox, std::vector<std::string>{"Point", "Face", "Geometry"})
        },
        {
            {gParamType_Geometry, "Output"}
        },
        {},
        {"geom"}
    });


    struct HasAttribute : INode {
        void apply() override {
            auto spGeo = get_input_Geometry("Input");
            auto attr_name = get_input2_string("attr_name");
            auto attrgroup = get_input2_string("Attribute Group");
            if (attr_name == "") {
                throw makeError<UnimplError>("the attribute name cannot be empty.");
            }

            bool hasAttr = false;
            if (attrgroup == "Point") {
                hasAttr = spGeo->has_point_attr(attr_name);
            }
            else if (attrgroup == "Face") {
                hasAttr = spGeo->has_face_attr(attr_name);
            }
            else if (attrgroup == "Geometry") {
                hasAttr = spGeo->has_face_attr(attr_name);
            }
            else {
                throw makeError<UnimplError>("unknown group");
            }
            set_output_int("hasAttr", hasAttr);
        }
    };
    ZENDEFNODE(HasAttribute,
    {
        {
            {gParamType_Geometry, "Input"},
            ParamPrimitive("attr_name", gParamType_String, "attribute1"),
            ParamPrimitive("Attribute Group", gParamType_String, "Point", zeno::Combobox, std::vector<std::string>{"Point", "Face", "Geometry"})
        },
        {
            ParamPrimitive("hasAttr", gParamType_Int)
        },
        {},
        {"geom"}
    });

    struct CopyAttributeFrom : INode {
        void apply() override {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            auto from_object = get_input_Geometry("From");
            auto source = get_input2_string("Source Attribute");
            auto target = get_input2_string("Target Attribute");
            std::string m_attrgroup = ZImpl(get_input2<std::string>("Attribute Group"));
            GeoAttrGroup group;
            if (m_attrgroup == "Point") {
                group = ATTR_POINT;
            }
            else if (m_attrgroup == "Face") {
                group = ATTR_FACE;
            }
            else if (m_attrgroup == "Geometry") {
                group = ATTR_GEO;
            }
            input_object->copy_attr_from(group, from_object, source, target);
            set_output("Output", std::move(input_object));
        }
    };
    ZENDEFNODE(CopyAttributeFrom,
    {
        {
            {gParamType_Geometry, "Input"},
            {gParamType_Geometry, "From"},
            ParamPrimitive("Source Attribute", gParamType_String, ""),
            ParamPrimitive("Target Attribute", gParamType_String, ""),
            ParamPrimitive("Attribute Group", gParamType_String, "Point", zeno::Combobox, std::vector<std::string>{"Point", "Face", "Geometry"})
        },
        {
            {gParamType_Geometry, "Output"}
        },
        {},
        {"geom"}
    });

    struct CopyAttribute : INode {
        void apply() override {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            auto source = get_input2_string("Source Attribute");
            auto target = get_input2_string("Target Attribute");
            std::string m_attrgroup = ZImpl(get_input2<std::string>("Attribute Group"));
            GeoAttrGroup group;
            if (m_attrgroup == "Point") {
                group = ATTR_POINT;
            }
            else if (m_attrgroup == "Face") {
                group = ATTR_FACE;
            }
            else if (m_attrgroup == "Geometry") {
                group = ATTR_GEO;
            }
            input_object->copy_attr(group, source, target);
            set_output("Output", std::move(input_object));
        }
    };
    ZENDEFNODE(CopyAttribute,
    {
        {
            {gParamType_Geometry, "Input"},
            ParamPrimitive("Source Attribute", gParamType_String, ""),
            ParamPrimitive("Target Attribute", gParamType_String, ""),
            ParamPrimitive("Attribute Group", gParamType_String, "Point", zeno::Combobox, std::vector<std::string>{"Point", "Face", "Geometry"})
        },
        {
            {gParamType_Geometry, "Output"}
        },
        {},
        {"geom"}
    });

    struct SetAttribute : INode {
        void apply() override {
            auto input_object = ZImpl(get_input2<GeometryObject_Adapter>("Input"));
            std::string attr_name = ZImpl(get_input2<std::string>("attr_name"));
            std::string m_type = ZImpl(get_input2<std::string>("Attribute Type"));
            std::string m_attrgroup = ZImpl(get_input2<std::string>("Attribute Group"));

            if (attr_name == "") {
                throw makeError<UnimplError>("the attribute name cannot be empty.");
            }

            AttrVar attr_value;
            if (m_type == "Vector2") {
                attr_value = ZImpl(get_input2<zeno::vec2f>("Vector2 Value"));
            }
            else if (m_type == "Vector3") {
                attr_value = ZImpl(get_input2<vec3f>("Vector3 Value"));
            }
            else if (m_type == "Vector4") {
                attr_value = ZImpl(get_input2<vec4f>("Vector4 Value"));
            }
            else if (m_type == "Integer") {
                attr_value = ZImpl(get_input2<int>("Integer Value"));
            }
            else if (m_type == "Float") {
                attr_value = ZImpl(get_input2<float>("Float Value"));
            }
            else if (m_type == "String") {
                attr_value = ZImpl(get_input2<std::string>("String Value"));
            }
            else if (m_type == "Boolean") {
                attr_value = ZImpl(get_input2<bool>("Boolean Value"));
            }

            if (auto spPrim = dynamic_cast<PrimitiveObject*>(input_object.get())) {
                throw makeError<UnimplError>("Unsupport Legacy Primitive Object for creating attribute");
            }
            else if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(input_object.get())) {
                if (m_attrgroup == "Point") {
                    if (!spGeo->m_impl->has_point_attr(attr_name)) {
                        throw makeError<UnimplError>("Input object does not have point attribute: " + attr_name);
                    }
                    spGeo->m_impl->set_point_attr(attr_name, attr_value);
                }
                else if (m_attrgroup == "Face") {
                    if (!spGeo->m_impl->has_face_attr(attr_name)) {
                        throw makeError<UnimplError>("Input object does not have face attribute: " + attr_name);
                    }
                    spGeo->m_impl->set_face_attr(attr_name, attr_value);
                }
                else if (m_attrgroup == "Geometry") {
                    if (!spGeo->m_impl->has_geometry_attr(attr_name)) {
                        throw makeError<UnimplError>("Input object does not have geometry attribute: " + attr_name);
                    }
                    spGeo->m_impl->set_geometry_attr(attr_name ,attr_value);
                }
            }
            else {
                throw makeError<UnimplError>("Unsupport Object for creating attribute");
            }
            ZImpl(set_output("Output", std::move(input_object)));
        }
    };
    ZENDEFNODE(SetAttribute,
    {
        {
            {gParamType_Geometry, "Input"},
            ParamPrimitive("attr_name", gParamType_String, "attribute1"),
            ParamPrimitive("Attribute Type",  gParamType_String, "float", zeno::Combobox, std::vector<std::string>{"Vector2", "Vector3", "Vector4", "Integer", "Float", "String", "Boolean"}),
            ParamPrimitive("Attribute Group", gParamType_String, "Point", zeno::Combobox, std::vector<std::string>{"Point", "Face", "Geometry"}),
            ParamPrimitive("Integer Value", gParamType_Int, 0, zeno::Lineedit, Any(), "visible = parameter('Attribute Type').value == 'Integer';"),
            ParamPrimitive("Float Value", gParamType_Float, 0, zeno::Lineedit, Any(), "visible = parameter('Attribute Type').value == 'Float';"),
            ParamPrimitive("String Value", gParamType_String, 0, zeno::Lineedit, Any(),"visible = parameter('Attribute Type').value == 'String';"),
            ParamPrimitive("Boolean Value", gParamType_Bool, 0, zeno::Checkbox, Any(),"visible = parameter('Attribute Type').value == 'Boolean';"),
            ParamPrimitive("Vector2 Value", gParamType_Vec2f, vec2f(0,0), zeno::Vec2edit, Any(),"visible = parameter('Attribute Type').value == 'Vector2';"),
            ParamPrimitive("Vector3 Value", gParamType_Vec3f, vec3f(0,0,0), zeno::Vec3edit, Any(),"visible = parameter('Attribute Type').value == 'Vector3';"),
            ParamPrimitive("Vector4 Value", gParamType_Vec4f, vec4f(0,0,0,0), zeno::Vec4edit, Any(),"visible = parameter('Attribute Type').value == 'Vector4';")
        },
        {
            {gParamType_Geometry, "Output"}
        },
        {},
        {"geom"}
    });
}

