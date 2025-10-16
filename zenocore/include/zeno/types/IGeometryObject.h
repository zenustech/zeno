#pragma once

#ifndef __IGEOMETRY_OBJECT_H__
#define __IGEOMETRY_OBJECT_H__

#include <set>
#include <zeno/core/coredata.h>
#include <zeno/core/common.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/api.h>

namespace zeno
{
    struct GeometryObject;  //private data.
    struct PrimitiveObject;

    using namespace zeno::reflect;

    struct ZENO_API GeometryObject_Adapter : IObject
    {
        zany clone() const override;
        void Delete() override;

        GeometryObject_Adapter();
        GeometryObject_Adapter(const GeometryObject_Adapter&) = delete;
        virtual ~GeometryObject_Adapter();

        void inheritAttributes(
            GeometryObject_Adapter* rhs,
            int vtx_offset,
            int pt_offset,
            std::set<zeno::String> pt_nocopy,    //TODO abi problem
            int face_offset,
            std::set<zeno::String> face_nocopy
        );

        std::vector<zeno::vec3f> points_pos();
        Vector<Vec3i> tri_indice() const;
        Vector<int> edge_list() const;
        void set_pos(int i, zeno::vec3f pos);
        //zeno::vec3f pos(int index) const;

        bool is_base_triangle() const;
        bool is_Line() const;
        int get_group_count(GeoAttrGroup grp) const;
        GeoAttrType get_attr_type(GeoAttrGroup grp, const zeno::String& name);
        zeno::Vector<zeno::String> get_attr_names(GeoAttrGroup grp);
        //void geomTriangulate(zeno::TriangulateInfo& info);

        //创建属性
        int create_attr(GeoAttrGroup grp, const zeno::String& attr_name, const Any& defl);
        int create_face_attr(const zeno::String& attr_name, const Any& defl);
        int create_point_attr(const zeno::String& attr_name, const Any& defl);
        int create_vertex_attr(const zeno::String& attr_name, const Any& defl);
        int create_geometry_attr(const zeno::String& attr_name, const Any& defl);
        std::vector<std::string> attributes(GeoAttrGroup grp);

        void copy_attr(GeoAttrGroup grp, const zeno::String& src_attr, const zeno::String& dest_attr);
        void copy_attr_from(GeoAttrGroup grp, GeometryObject_Adapter* pSrcObject, const zeno::String& src_attr, const zeno::String& dest_attr);

        //获取属性
        std::vector<float> get_float_attr(GeoAttrGroup grp, const zeno::String& attr_name);
        std::vector<vec3f> get_vec3f_attr(GeoAttrGroup grp, const zeno::String& attr_name);

        //设置属性
        int set_attr(GeoAttrGroup grp, const zeno::String& name, const Any& val);
        int set_vertex_attr(const zeno::String& attr_name, const Any& defl);
        int set_point_attr(const zeno::String& attr_name, const Any& defl);
        int set_face_attr(const zeno::String& attr_name, const Any& defl);
        int set_geometry_attr(const zeno::String& attr_name, const Any& defl);

        void foreach_vec3_attr_update(GeoAttrGroup grp, const zeno::String& attr_name, char channel, std::function<zeno::vec3f(int idx, zeno::vec3f old_elem_value)>&& evalf);
        void foreach_float_attr_update(GeoAttrGroup grp, const zeno::String& attr_name, char channel, std::function<float(int idx, float old_elem_value)>&& evalf);

        /* 检查属性是否存在 */
        bool has_attr(GeoAttrGroup grp, const zeno::String& name, GeoAttrType type = ATTR_TYPE_UNKNOWN);
        bool has_vertex_attr(const zeno::String& name) const;
        bool has_point_attr(const zeno::String& name) const;
        bool has_face_attr(const zeno::String& name) const;
        bool has_geometry_attr(const zeno::String& name) const;

        //删除属性
        int delete_attr(GeoAttrGroup grp, const zeno::String& attr_name);
        int delete_vertex_attr(const zeno::String& attr_name);
        int delete_point_attr(const zeno::String& attr_name);
        int delete_face_attr(const zeno::String& attr_name);
        int delete_geometry_attr(const zeno::String& attr_name);

        /* 添加元素 */
        int add_vertex(int face_id, int point_id);
        int add_point(Vec3f pos);
        int add_face(const zeno::Vector<int>& points, bool bClose = true);
        void set_face(int idx, const zeno::Vector<int>& points, bool bClose = true);

        /* 移除元素相关 */
        bool remove_faces(const std::set<int>& faces, bool includePoints);
        bool remove_point(int ptnum);
        bool remove_vertex(int face_id, int vert_id);

        /* 返回元素个数 */
        int npoints() const;
        int nfaces() const;
        int nvertices() const;
        int nvertices(int face_id) const;
        int nattributes(GeoAttrGroup grp) const;

        /* 点相关 */
        zeno::Vector<int> point_faces(int point_id);
        int point_vertex(int point_id);
        zeno::Vector<int> point_vertices(int point_id);

        /* 面相关 */
        int face_point(int face_id, int vert_id) const;
        zeno::Vector<int> face_points(int face_id);
        int face_vertex(int face_id, int vert_id);
        int face_vertex_count(int face_id);
        zeno::Vector<int> face_vertices(int face_id);
        zeno::Vec3f face_normal(int face_id);

        /* Vertex相关 */
        int vertex_index(int face_id, int vertex_id);
        int vertex_next(int linear_vertex_id);
        int vertex_prev(int linear_vertex_id);
        int vertex_point(int linear_vertex_id);
        int vertex_face(int linear_vertex_id);
        int vertex_face_index(int linear_vertex_id);
        std::tuple<int, int, int> vertex_info(int linear_vertex_id);

        std::unique_ptr<PrimitiveObject> toPrimitiveObject() const;
        std::unique_ptr<GeometryObject_Adapter> toIndiceMeshesTopo() const;
        std::unique_ptr<GeometryObject_Adapter> toHalfEdgeTopo() const;

        std::unique_ptr<GeometryObject> m_impl;
    };

    //没有放到外面，是因为用户要直接include当前文件，干脆直接在这里构造
    ZENO_API std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(GeomTopoType type);
    ZENO_API std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(GeomTopoType type, bool bTriangle, int nPoints, int nFaces, bool bInitFaces = false);
    ZENO_API std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(
        GeomTopoType type,
        bool bTriangle,
        const std::vector<zeno::vec3f>& points,
        const std::vector<std::vector<int>>& faces);
    ZENO_API std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(PrimitiveObject* prim);
    ZENO_API std::unique_ptr<GeometryObject_Adapter> clone_GeometryObject(std::unique_ptr<GeometryObject_Adapter> pGeom);
}

#endif