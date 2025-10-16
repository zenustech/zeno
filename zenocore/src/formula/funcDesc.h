#include <zeno/core/data.h>

namespace zeno
{
    namespace zfx
    {
        std::map<std::string, FUNC_INFO> funcsDesc = {
            {"sin",
                {"sin",
                "Return the sine of the argument.",
                "float",
                {{"degree", "float"}}
                }
            },
            {"cos",
                {"cos",
                "Return the cose of the argument.",
                "float",
                { {"degree", "float"}}}
            },
            {"sinh",
                {"sinh",
                "Return the hyperbolic sine of the argument.",
                "float",
                { {"number", "float"}}}
            },
            {"cosh",
                {"cosh",
                "Return the hyperbolic cose of the argument.",
                "float",
                { {"number", "float"}}}
            },
            {"ref",
                {"ref",
                "Return the value of reference param of node.\n"\
                "path-to-param:\n    Reference path of parameter.",
                "float",
                { {"path-to-param", "string"}}}
            },
            {"rand",
                {"rand",
                "Returns a pseudo-number number from 0 to 1.",
                "float", {{"seed", "int"}}}
            },
            {"pow",
                {"pow",
                "Find first param raised to the second param power.",
                "float", {{"number", "float"}, {"exponent", "float"}}}
            },
            {"param",
                {"param",
                "Get parameter from the previous node.",
                "primType/ObjType", {{"param-name", "string"}}}
            },
            {"log",
                {"log",
                "Print variables.",
                "void", {{"logFormat", "string"}, {"arg1", "any"}, {"arg2...", "any"}}}
            },
            {"vec3",
                {"vec3",
                "Create vec3 variable.",
                "vec3", {{"x", "float"}, {"y", "float"}, {"z", "float"}}}
            },
            {"create_attr",
                {"create_attr",
                "Adds an attribute to a geometry, return -1 on failure. \n"\
                "group:\n    One of \"vertex\",\"point\",\"face\",\"geometry.\"\n"\
                "name:\n    The name of the attribute to create.\n"\
                "value:\n    The default value for the attribute and determines the type of attribute to create.",
                "int", {{"group", "string"}, {"name", "string"}, {"value", "any"}}}
            },
            {"create_face_attr",
                {"create_face_attr",
                "Adds a face attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to create.\n"\
                "value:\n    The default value for the attribute and determines the type of attribute to create.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            {"create_point_attr",
                {"create_point_attr",
                "Adds a point attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to create.\n"\
                "value:\n    The default value for the attribute and determines the type of attribute to create.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            {"create_vertex_attr",
                {"create_vertex_attr",
                "Adds a vertex attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to create.\n"\
                "value:\n    The default value for the attribute and determines the type of attribute to create.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            {"create_geometry_attr",
                {"create_geometry_attr",
                "Adds a geometry attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to create.\n"\
                "value:\n    The default value for the attribute and determines the type of attribute to create.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            {"set_attr",
                {"set_attr",
                "Writes an attribute to a geometry, return -1 on failure. \n"\
                "group:\n    One of \"vertex\",\"point\",\"face\",\"geometry.\"\n"\
                "name:\n    The name of the attribute to change.\n"\
                "value:\n    The value to set, and determines the type of attribute to set.",
                "int", { {"group", "string"}, {"name", "string"}, {"value", "any"} }}
            },
            {"set_vertex_attr",
                {"set_vertex_attr",
                "Writes a vertex attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to change.\n"\
                "value:\n    The value to set, and determines the type of attribute to set.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            {"set_point_attr",
                {"set_point_attr",
                "Writes a point attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to change.\n"\
                "value:\n    The value to set, and determines the type of attribute to set.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            { "set_face_attr",
                {"set_face_attr",
                "Writes a face attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to change.\n"\
                "value:\n    The value to set, and determines the type of attribute to set.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            {"set_geometry_attr",
                {"set_geometry_attr",
                "Writes a geometry attribute to a geometry, return -1 on failure. \n"\
                "name:\n    The name of the attribute to change.\n"\
                "value:\n    The value to set, and determines the type of attribute to set.",
                "int", {{"name", "string"}, {"value", "any"}}}
            },
            { "get_attr",
                {"get_attr",
                "Get attribute from a geometry. \n"\
                "group:\n    One of \"vertex\",\"point\",\"face\",\"geometry.\"\n"\
                "name:\n    The name of the attribute to change.",
                "int", { {"group", "string"}, {"name", "string"} }}
            },
            { "get_vertex_attr",
                {"get_vertex_attr",
                "Get vertex attribute from a geometry. \n"\
                "name:\n    The name of the attribute to get.",
                "int", {{"name", "string"}}}
            },
            { "get_point_attr",
                {"get_point_attr",
                "Get point attribute from a geometry. \n"\
                "name:\n    The name of the attribute to get.",
                "int", {{"name", "string"}}}
            },
            { "get_face_attr",
                {"get_face_attr",
                "Get face attribute from a geometry. \n"\
                "name:\n    The name of the attribute to get.",
                "int", {{"name", "string"}}}
            },
            { "get_geometry_attr",
                {"get_geometry_attr",
                "Get geometry attribute from a geometry. \n"\
                "name:\n    The name of the attribute to get.",
                "int", {{"name", "string"}}}
            },
            { "has_attr",
                {"has_attr",
                "Checks whether an attribute exists. \n"\
                "group:\n    One of \"vertex\",\"point\",\"face\",\"geometry.\"\n"\
                "name:\n    The name of the attribute to check.",
                "bool", {{"group", "string"}, {"name", "string"}}}
            },
            { "has_vertex_attr",
                { "has_vertex_attr",
                "Checks whether a vertex attribute exists. \n"\
                "name:\n    The name of the vertex attribute to check.",
                "bool", {{"name", "string"}} }
            },
            { "has_point_attr",
                { "has_point_attr",
                "Checks whether a point attribute exists. \n"\
                "name:\n    The name of the point attribute to check.\n",
                "bool", {{"name", "string"}} }
            },
            { "has_face_attr",
                { "has_face_attr",
                "Checks whether a face attribute exists. \n"\
                "name:\n    The name of the face attribute to check.\n",
                "bool", {{"name", "string"}} }
            },
            { "has_geometry_attr",
                { "has_geometry_attr",
                "Checks whether a geometry attribute exists. \n"\
                "name:\n    The name of the geometry attribute to check.\n",
                "bool", {{"name", "string"}} }
            },
            { "delete_attr",
                { "delete_attr",
                "Removes an attribute, return 0 on failure.\n"\
                "group:\n    One of \"vertex\",\"point\",\"face\",\"geometry.\"\n"\
                "name:\n    The name of the attribute to remove.\n",
                "int", {{"group", "string"}, {"name", "string"}}}
            },
            { "delete_vertex_attr",
                { "delete_vertex_attr",
                "Removes a vertex attribute, return 0 on failure.\n"\
                "name:\n    The name of the attribute to remove.\n",
                "int", { {"name", "string"} } }
            },
            { "delete_point_attr",
                { "delete_point_attr",
                "Removes a point attribute, return 0 on failure.\n"\
                "name:\n    The name of the attribute to remove.\n",
                "int", {{"name", "string"}} }
            },
            { "delete_face_attr",
                { "delete_face_attr",
                "Removes a face attribute, return 0 on failure.\n"\
                "name:\n    The name of the attribute to remove.\n",
                "int", {{"name", "string"}} }
            },
            { "delete_geometry_attr",
                { "delete_geometry_attr",
                "Removes a geometry attribute, return 0 on failure.\n"\
                "name:\n    The name of the attribute to remove.\n",
                "int", {{"name", "string"}} }
            },
            { "add_vertex",
                { "add_vertex",
                "Adds a vertex to a face in a geometry, return vertex id in face.\n"\
                "faceId:\n    The face number to add the vertex to.\n"\
                "pointId:\n    The point number to wire the new vertex to.",
                "int", {{"faceId", "int"}, {"pointId", "int"}}}
            },
            { "add_point",
                { "add_point",
                "Adds a point to the geometry, return point number, -1 on failure.\n"\
                "pos:\n    The point position.",
                "int", {{"pos", "vec3f"}} }
            },
            { "add_face",
                { "add_face",
                "Adds a face to the geometry, return face number, -1 on failure.\n"\
                "points:\n    The vector of pointId.",
                "int", {{"points", "vector<int>"}} }
            },
            { "remove_face",
                { "remove_face",
                "Remove faces from the geometry.\n"\
                "faces:\n    The vector of pointId.\n"\
                "includePoints:\n    If true, will also remove any points associated with the face that are not associated with any other faces.",
                "bool", {{"remFace", "int"}, {"includePoints", "bool"}}}
            },
            { "remove_point",
                { "remove_point",
                "Remove a point from the geometry.\n"\
                "pointId:\n    Id of the point to be removed.",
                "bool", {{"pointId", "int"}}}
            },
            { "remove_vertex",
                { "remove_vertex",
                "Remove a vertex from the geometry.\n"\
                "faceId:\n    Id of face that contains the vertex to be removed."\
                "vertexId:\n    Id of the vertex in face.",
                "bool", {{"faceId", "int"}, {"vertexId", "int"}}}
            },
            { "npoints",
                { "npoints",
                "Returns the number of points in the input or geometry.\n"\
                "path-to-obj(optional):\n    Reference path of obj",
                "int", { {"path-to-obj", "string"} } }
            },
            { "nfaces",
                { "nfaces",
                "Returns the number of faces in the input or geometry.",
                "int", {} }
            },
            { "nvertices",
                { "nvertices",
                "Returns the number of vertices in the input or geometry.",
                "int", {} }
            },
            { "point_faces",
                { "point_faces",
                "Returns the list of faces containing a point.\n"\
                "pointId:\n    Id of the point.",
                "vector", {{"pointId", "int"}}}
            },
            { "point_vertex",
                { "point_vertex",
                "Returns a linear vertex number of a point in a geometry.\n"
                "pointId:\n    Id of the point.",
                "int", {{"pointId", "int"}}}
            },
            { "point_vertices",
                { "point_vertices",
                "Returns the list of linear vertex indices connected to a point.\n"\
                "pointId:\n    Id of the point.",
                "int", {{"pointId", "int"}} }
            },
            { "face_point",
                { "face_point",
                "Converts a face/vertex pair into a point number."\
                "faceId:\n    Id of the face.\n"\
                "vertexId:\n    Id of the vertex in face.",
                "int", {{"faceId", "int"}, {"vertexId", "int"}}}
            },
            { "face_points",
                { "face_points",
                "Returns the list of points on a face.\n"
                "faceId:\n    Id of the face.",
                "vector", {{"faceId", "int"}}}
            },
            { "face_vertex",
                { "face_vertex",
                "Converts a face/vertex pair into a linear vertex.\n"\
                "faceId:\n    Id of the face.\n"\
                "vertexId:\n    Id of the vertex in face.",
                "int", {{"faceId", "int"}, {"vertexId", "int"}} }
            },
            { "face_vertex_count",
                { "face_vertex_count",
                "Returns number of vertices in a face in a geometry.\n"\
                "faceId:\n    Id of the face.",
                "int", { {"faceId", "int"} } }
            },
            { "face_vertices",
                { "face_vertices",
                "Returns the list of linear vertex indices on a face.\n"\
                "faceId:\n    Id of the face.",
                "vector", {{"faceId", "int"}} }
            },
            { "vertex_index",
                { "vertex_index",
                "Converts a face/vertex pair into a linear vertex.\n"\
                "faceId:\n    Id of the face.\n"\
                "vertexId:\n    Id of the vertex in face.",
                "int", {{"faceId", "int"}, {"vertexId", "int"}} }
            },
            { "vertex_next",
                { "vertex_next",
                "Returns the linear vertex number of the next vertex sharing a point with a given vertex.\n"\
                "linearVertex:\n    Linear Vertex number of vertex",
                "int", {{"linearVertex", "int"}} }
            },
            { "vertex_prev",
                { "vertex_prev",
                "Returns the linear vertex number of the previous vertex sharing a point with a given vertex.\n"\
                "linearVertex:\n    Linear Vertex number of vertex",
                "int", {{"linearVertex", "int"}} }
            },
            { "vertex_point",
                { "vertex_point",
                "Returns the point number of linear vertex in a geometry.\n"\
                "linearVertex:\n    Linear Vertex number of vertex",
                "int", {{"linearVertex", "int"}} }
            },
            { "vertex_face",
                { "vertex_face",
                "Returns the number of the face containing a given vertex.\n"\
                "linearVertex:\n    Linear Vertex number of vertex",
                "int", {{"linearVertex", "int"}} }
            },
            { "vertex_face_index",
                { "vertex_face_index",
                "Converts a linear vertex index into a face vertex number.\n"\
                "linearVertex:\n    Linear Vertex number of vertex",
                "int", {{"linearVertex", "int"}} }
            },
            { "bbox",
                { "bbox",
                "Returns bounding box information for a surface node.\n"\
                "path-to-node:\n    Reference path of node\n"\
                "type:\n    can be one of D_XMIN, D_YMIN, D_ZMIN, D_XMAX, D_YMAX, D_ZMAX, D_XSIZE, D_YSIZE, or D_ZSIZE for the corresponding values of the bounding box, return -1 if there is no size attribute.",
                "int", {{"size value", "int"}} }
            },
            { "rint",
                { "rint", "i.e. rint(2.2) = 2; rint(2.6) = 3;",
                {{"float"}}
                }
            },
            { "pcopen",
                {"pcopen", "", "int", {
                    {"input geometry", "string"},
                    {"attr name", "string"},
                    {"vector P", "vector"},
                    {"radius", "float"},
                    {"maxpoints", "int"}
                }}
            },
            {
                "pcnumfound",
                {
                    "pcnumfound", "", "int", {
                        {"handle", "int"}
                    }
                }
            },
            {
                "append",
                {
                    "append", "append content into array", "array", {
                        {"array", "T[]"},
                        {"element", "T"}
                    }
                }
            },
            {
                "len",
                {
                    "len", "get the length of an array", "int", {
                        {"array", "T[]"}
                    }
                }
            },
            {
                "fit",
                {
                    "fit", "Takes the value in one range and shifts it to the corresponding value in a new range. e.g. fit(.3, 0, 1, 10, 20) == 13",
                    "float", {
                        {"value", "float"},
                        {"omin", "float"},
                        {"omax", "float"},
                        {"nmin", "float"},
                        {"nmax", "float"}
                    }
                }
            },
            {
                "fit01",
                {
                    "fit01", "Returns a number between newmin and newmax that is relative to num in the range between 0 and 1. If the value is outside the 0 to 1 it will be clamped to the new range.",
                    "float", {
                        {"value", "float"},
                        {"nmin", "float"},
                        {"nmax", "float"}
                    }
                }
            },
            {
                "get_bboxmin",
                {
                    "get_bboxmin", "Returns the min vec3 of the bbox of the geometry",
                    "vec3", {}
                }
            },
            {
                "get_bboxmax",
                {
                    "get_bboxmax", "Returns the max vec3 of the bbox of the geometry",
                    "vec3", {}
                }
            }
        };
    }
}