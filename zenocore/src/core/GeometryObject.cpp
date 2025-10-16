#include "geotopology.h"
#include <zeno/types/GeometryObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <assert.h>
#include <zeno/formula/syntax_tree.h>
#include <zeno/utils/vectorutil.h>
#include <zeno/utils/format.h>
#include "../utils/zfxutil.h"
#include "zeno_types/reflect/reflection.generated.hpp"
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/core/Session.h>
#include <zeno/core/ObjectRecorder.h>
#include <regex>


namespace zeno
{
    //#define TEST_GEOMETRY_LEAK

    template <class T>
    static T get_zfxvar(zfxvariant value) {
        return std::visit([](auto const& val) -> T {
            using V = std::decay_t<decltype(val)>;
            if constexpr (!std::is_constructible_v<T, V>) {
                throw makeError<TypeError>(typeid(T), typeid(V), "get<zfxvariant>");
            }
            else {
                return T(val);
            }
            }, value);
    }

    GeometryObject::GeometryObject(GeomTopoType type)
        : m_type(type)
    {
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geom_impls.insert(this);
#endif
    }

    GeometryObject::GeometryObject(GeomTopoType type, bool bTriangle, int nPoints, int nFaces, bool bInitFaces)
        : m_type(type)
    {
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geom_impls.insert(this);
#endif
        if (Topo_IndiceMesh == type) {
            m_spTopology = create_indicemesh_topo(bTriangle, nPoints, nFaces, bInitFaces);
        }
        else if (Topo_HalfEdge == type) {
            m_spTopology = create_halfedge_topo(bTriangle, nPoints, nFaces, bInitFaces);
        }
        else {
            throw makeError<UnimplError>("unknown type of Geometry Topology");
        }
    }

    GeometryObject::GeometryObject(GeomTopoType type, bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces)
        : m_type(type)
    {
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geom_impls.insert(this);
#endif
        if (Topo_IndiceMesh == type) {
            m_spTopology = create_indicemesh_topo(bTriangle, nPoints, faces);
        }
        else if (Topo_HalfEdge == type) {
            m_spTopology = create_halfedge_topo(bTriangle, nPoints, faces);
        }
        else {
            throw makeError<UnimplError>("unknown type of Geometry Topology");
        }
    }

    GeometryObject::GeometryObject(PrimitiveObject* spPrim)
        : m_type(Topo_IndiceMesh)
    {
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geom_impls.insert(this);
#endif
        m_spTopology = create_indicemesh_topo(spPrim);
        //提取出prim所有的属性
        create_attr(ATTR_POINT, "pos", spPrim->verts.values);
        for (auto& [attr_name, var_vec] : spPrim->verts.attrs) {
            create_attr_from_AttrVector(ATTR_POINT, attr_name, var_vec);
        }
        //面属性
        for (auto& [attr_name, var_vec] : spPrim->polys.attrs) {
            create_attr_from_AttrVector(ATTR_FACE, attr_name, var_vec);
        }
        for (auto& [attr_name, var_vec] : spPrim->tris.attrs) {
            create_attr_from_AttrVector(ATTR_FACE, attr_name, var_vec);
        }
    }

    GeometryObject::GeometryObject(const GeometryObject& rhs)
        : m_spTopology(rhs.m_spTopology)
        , m_type(rhs.m_type)
    {
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geom_impls.insert(this);
#endif
        m_vert_attrs = rhs.m_vert_attrs;
        m_point_attrs = rhs.m_point_attrs;
        m_face_attrs = rhs.m_face_attrs;
        m_geo_attrs = rhs.m_geo_attrs;
    }

    GeometryObject::~GeometryObject() {
        int usecnt = m_spTopology.use_count();
        if (usecnt > 0) {
            int j;
            j = 0;
        }
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geom_impls.erase(this);
#endif
    }

    void GeometryObject::_temp_code_regist() {
        
    }

    void GeometryObject::_temp_code_unregist() {

    }

    GeomTopoType GeometryObject::type() const {
        return m_type;
    }

    std::unique_ptr<PrimitiveObject> GeometryObject::toPrimitive() {
        auto spPrim = std::make_unique<PrimitiveObject>();
        std::vector<vec3f> vec_pos = points_pos();
        int nPoints = m_spTopology->npoints();
        assert(nPoints == vec_pos.size());
        spPrim->verts.resize(nPoints);
        for (int i = 0; i < nPoints; i++) {
            spPrim->verts[i] = vec_pos[i];
        }

        for (auto& [name, sp_attr_data] : m_point_attrs) {
            if (name == "pos") {
                continue;
            }
            sp_attr_data.to_prim_attr(spPrim.get(), true, false, name);
        }

        for (auto& [name, sp_attr_data] : m_face_attrs) {
            sp_attr_data.to_prim_attr(spPrim.get(), false, m_spTopology->is_base_triangle(), name);
        }

        auto primTopo = get_primitive_topo(m_spTopology);
        //拷拓扑就行
        spPrim->lines.values = primTopo->lines.values;
        spPrim->tris.values = primTopo->tris.values;
        spPrim->quads.values = primTopo->quads.values;
        spPrim->loops.values = primTopo->loops.values;
        spPrim->polys.values = primTopo->polys.values;
        spPrim->edges.values = primTopo->edges.values;
        return spPrim;
    }

    static AttrVar get_init_value(GeoAttrType type) {
        AttrVar init_val;
        switch (type)
        {
        case ATTR_INT:      init_val = 0; break;
        case ATTR_FLOAT:    init_val = 0.f; break;
        case ATTR_STRING:   init_val = ""; break;
        case ATTR_VEC2:     init_val = vec2f(); break;
        case ATTR_VEC3:     init_val = vec3f(); break;
        case ATTR_VEC4:     init_val = vec4f(); break;
        }
        return init_val;
    }

    void GeometryObject::inheritAttributes(
        GeometryObject* rhs,
        int vtx_offset,
        int pt_offset,
        std::set<std::string> pt_nocopy,
        int face_offset,
        std::set<std::string> face_nocopy)
    {
        //这里假定了点和面是确定了的
        //点属性
        if (pt_offset != -1) {
            for (const auto& [name, attr_vec] : rhs->m_point_attrs) {
                if (pt_nocopy.find(name) != pt_nocopy.end())
                    continue;
                auto iter = m_point_attrs.find(name);
                if (iter == m_point_attrs.end()) {
                    GeoAttrType type = attr_vec.type();
                    AttrVar init_val = get_init_value(type);
                    create_point_attr(name, init_val);
                    iter = m_point_attrs.find(name);
                }
                iter->second.copySlice(attr_vec, pt_offset);
            }
        }
        if (face_offset != -1) {
            //必须要在addface以后，否则大小还没定下来
            for (const auto& [name, attr_vec] : rhs->m_face_attrs) {
                auto iter = m_face_attrs.find(name);
                if (face_nocopy.find(name) != face_nocopy.end())
                    continue;
                if (iter == m_face_attrs.end()) {
                    GeoAttrType type = attr_vec.type();
                    AttrValue val = attr_vec.front();
                    AttrVar init_val = get_init_value(type);
                    create_face_attr(name, init_val);
                    iter = m_face_attrs.find(name);
                }
                iter->second.copySlice(attr_vec, face_offset);
            }
        }
    }

    void GeometryObject::create_attr_from_AttrVector(GeoAttrGroup grp, const std::string& attr_name, const AttrVectorVariant& var_vec) {
        std::visit([&](auto&& vec) {
            using T = std::decay_t<decltype(vec)>;
            if constexpr (std::is_same_v<T, std::vector<vec3f>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<float>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<vec3i>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<int>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<vec2f>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<vec2i>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<vec4f>>) {
                create_attr(grp, attr_name, vec);
            }
            else if constexpr (std::is_same_v<T, std::vector<vec4i>>) {
                create_attr(grp, attr_name, vec);
            }
        }, var_vec);
    }

    void GeometryObject::initFromPrim(PrimitiveObject* prim) {
        //不考虑points，lines, quads, edges, mtl. inst，遇到再处理
        if (!prim->points->empty() || !prim->lines->empty() || !prim->edges->empty()
            || !prim->quads->empty()) {
            throw makeError<UnimplError>("cannot wrap primitive Object by Geom for points, lines, edges and quads");
        }

        //现在有bind机制，也许不需要再这么麻烦地转来转去
        create_attr(ATTR_POINT, "pos", prim->verts.values);

        m_spTopology = create_indicemesh_topo(prim);

        //顶点属性
        for (auto& [attr_name, var_vec] : prim->verts.attrs) {
            create_attr_from_AttrVector(ATTR_POINT, attr_name, var_vec);
        }

        //面属性
        for (auto& [attr_name, var_vec] : prim->polys.attrs) {
            create_attr_from_AttrVector(ATTR_FACE, attr_name, var_vec);
        }
        for (auto& [attr_name, var_vec] : prim->tris.attrs) {
            create_attr_from_AttrVector(ATTR_FACE, attr_name, var_vec);
        }
    }

    std::unique_ptr<GeometryObject> GeometryObject::toIndiceMeshesTopo() const {
        if (!m_spTopology) return nullptr;

        zeno::GeomTopoType type = m_spTopology->type();
        if (zeno::Topo_IndiceMesh == type) {
            return std::make_unique<GeometryObject>(*this);
        }
        else {
            auto pGeom = std::make_unique<GeometryObject>(Topo_IndiceMesh);
            pGeom->m_spTopology = create_indicemesh_by_halfedge(m_spTopology);
            pGeom->m_point_attrs = m_point_attrs;
            pGeom->m_vert_attrs = m_vert_attrs;
            pGeom->m_face_attrs = m_face_attrs;
            pGeom->m_geo_attrs = m_geo_attrs;
            return pGeom;
        }
    }

    std::unique_ptr<GeometryObject> GeometryObject::toHalfEdgeTopo() const {
        if (!m_spTopology) return nullptr;

        zeno::GeomTopoType type = m_spTopology->type();
        if (zeno::Topo_IndiceMesh == type) {
            auto pGeom = std::make_unique<GeometryObject>(Topo_HalfEdge);
            pGeom->m_spTopology = create_halfedge_by_indicemesh(npoints(), m_spTopology);
            pGeom->m_point_attrs = m_point_attrs;
            pGeom->m_vert_attrs = m_vert_attrs;
            pGeom->m_face_attrs = m_face_attrs;
            pGeom->m_geo_attrs = m_geo_attrs;
            return pGeom;
        }
        else {
            return std::make_unique<GeometryObject>(*this);
        }
    }

    std::vector<vec3f> GeometryObject::points_pos() {
        std::map<std::string, AttributeVector>& container = get_container(ATTR_POINT);
        auto iter = container.find("pos");
        if (iter == container.end()) {
            throw makeError<KeyError>("pos", "not exist on point attr");
        }
        return iter->second.get_attrs<vec3f>();
    }

    std::vector<vec3i> GeometryObject::tri_indice() const {
        return m_spTopology->tri_indice();
    }

    std::vector<int> GeometryObject::edge_list() const {
        return m_spTopology->edge_list();
    }

    std::vector<std::vector<int>> GeometryObject::face_indice() const {
        return m_spTopology->face_indice();
    }

    bool GeometryObject::is_base_triangle() const {
        return m_spTopology->is_base_triangle();
    }

    bool GeometryObject::is_Line() const {
        return m_spTopology->is_line();
    }

    int GeometryObject::get_group_count(GeoAttrGroup grp) const {
        switch (grp) {
        case ATTR_POINT: return m_spTopology->npoints();
        case ATTR_FACE: return m_spTopology->nfaces();
        case ATTR_VERTEX: return m_spTopology->nvertices();
        case ATTR_GEO: return 1;
        default:
            return 0;
        }
    }

    void GeometryObject::geomTriangulate(zeno::TriangulateInfo& info) {
        if (is_base_triangle()) {
            //TODO
            return;
        }
        //m_spTopology->geomTriangulate(info);
        //TODO: uv
    }

    bool GeometryObject::remove_point(int ptnum) {
        std::vector<vec3f> vec_pos = points_pos();
        assert(m_spTopology->npoints() == vec_pos.size());

        if (m_vert_attrs.size() > 0) {
            auto firstIter = m_vert_attrs.begin();
            assert(firstIter->second.size() == m_spTopology->nvertices());
        }

        copyTopologyAccordtoUseCount();
        std::vector<int> linearVertexIdx = m_spTopology->point_vertices(ptnum);
        std::vector<int> facesIdx = m_spTopology->point_faces(ptnum);

        bool ret = m_spTopology->remove_point(ptnum);
        if (ret) {
            //属性也要同步删除,包括point和vertices
            for (auto& [name, attrib_vec] : m_point_attrs) {
                removeAttribElem(attrib_vec, ptnum);
            }

            for (auto& [name, attrib_vec] : m_vert_attrs) {
                for (int i = linearVertexIdx.size() - 1; i >= 0; i--) {
                    removeAttribElem(attrib_vec, linearVertexIdx[i]);
                }
            }

            for (auto& [name, attrib_vec] : m_face_attrs) {
                for (int i = facesIdx.size() - 1; i >= 0; i--) {
                    removeAttribElem(attrib_vec, facesIdx[i]);
                }
            }

            CALLBACK_NOTIFY(remove_point, ptnum)
            CALLBACK_NOTIFY(reset_faces)
            CALLBACK_NOTIFY(reset_vertices)
        }
        return ret;
    }

    bool GeometryObject::remove_vertex(int face_id, int vert_id) {
        copyTopologyAccordtoUseCount();

        size_t linear_vertex = m_spTopology->vertex_index(face_id, vert_id);
        int vertexCount = m_spTopology->face_vertex_count(face_id);

        bool ret = m_spTopology->remove_vertex(face_id, vert_id);
        if (ret) {
            for (auto& [name, attrib_vec] : m_vert_attrs) {
                removeAttribElem(attrib_vec, linear_vertex);
            }
            for (auto& [name, attrib_vec] : m_face_attrs) {
                removeAttribElem(attrib_vec, face_id);
            }
            if (m_spTopology->is_base_triangle() || vertexCount == 3) {//如果有3个顶点删除一个也要删除face
                CALLBACK_NOTIFY(reset_vertices)
                CALLBACK_NOTIFY(remove_face, face_id)
            } else {
                CALLBACK_NOTIFY(remove_vertex, linear_vertex)
            }
            return true;
        } else {
            return false;
        }
    }

    //给定 face_id 和 vert_id，返回顶点索引编号 point_idx。
    int GeometryObject::face_point(int face_id, int vert_id) const
    {
        return m_spTopology->face_point(face_id, vert_id);
    }

    //通过 face_id，获取此 face 所有 points 索引编号。
    std::vector<int> GeometryObject::face_points(int face_id)
    {
        return m_spTopology->face_points(face_id);
    }

    //通过 face_id和内部的vertex_id索引，返回其linearindex.
    int GeometryObject::face_vertex(int face_id, int vert_id)
    {
        return m_spTopology->face_vertex(face_id, vert_id);
    }

    int GeometryObject::face_vertex_count(int face_id)
    {

        return m_spTopology->face_vertex_count(face_id);
    }

    std::vector<int> GeometryObject::face_vertices(int face_id)
    {

        return m_spTopology->face_vertices(face_id);
    }

    zeno::vec3f GeometryObject::face_normal(int face_id) {

        const std::vector<int>& pts = face_points(face_id);
        std::vector<vec3f> pos = points_pos();
        if (pts.size() > 2) {
            vec3f v12 = pos[pts[1]] - pos[pts[0]];
            vec3f v23 = pos[pts[2]] - pos[pts[1]];
            return zeno::normalize(zeno::cross(v12, v23));
        }
        return zeno::vec3f();
    }

    //返回包含指定 point 的 face 列表。
    std::vector<int> GeometryObject::point_faces(int point_id)
    {
        return m_spTopology->point_faces(point_id);
    }

    /*
        Returns the linear vertex number of the first vertex to share this point.
        Returns -1 if no vertices share this point.
    */
    int GeometryObject::point_vertex(int point_id)
    {
        return m_spTopology->point_vertex(point_id);
    }

    /*
        An array of linear vertices that are wired to the given point. You should not rely on the numbers being in a particular order.

        If the given point contains no vertices, the array will be empty.
     */
    std::vector<int> GeometryObject::point_vertices(int point_id)
    {
        return m_spTopology->point_vertices(point_id);
    }

    size_t GeometryObject::get_attr_size(GeoAttrGroup grp) const {
        if (grp == ATTR_GEO) {
            return 1;
        }
        else if (grp == ATTR_POINT) {
            return m_spTopology->npoints();
        }
        else if (grp == ATTR_FACE) {
            return m_spTopology->nfaces();
        }
        else if (grp == ATTR_VERTEX) {
            return m_spTopology->nvertices();
        }
        else {
            throw makeError<UnimplError>("Unknown group on attr");
        }
    }

    void GeometryObject::copyTopologyAccordtoUseCount() {
        if (m_spTopology.use_count() > 1) {
            m_spTopology = clone_topology(m_spTopology);
        }
    }

    void GeometryObject::removeAttribElem(AttributeVector& attrib_vec, int idx)
    {
        GeoAttrType attrType = attrib_vec.type();
        if (attrType == ATTR_INT) {
            attrib_vec.remove_elem<int>(idx);
        }
        else if (attrType == ATTR_FLOAT) {
            attrib_vec.remove_elem<float>(idx);
        }
        else if (attrType == ATTR_STRING) {
            attrib_vec.remove_elem < std::string >(idx);
        }
        else if (attrType == ATTR_VEC2) {
            attrib_vec.remove_elem<zeno::vec2f>(idx);
        }
        else if (attrType == ATTR_VEC3) {
            attrib_vec.remove_elem<zeno::vec3f>(idx);
        }
        else if (attrType == ATTR_VEC4) {
            attrib_vec.remove_elem<zeno::vec4f>(idx);
        }
    }

    std::map<std::string, AttributeVector>& GeometryObject::get_container(GeoAttrGroup grp) {
        if (grp == ATTR_GEO) {
            return m_geo_attrs;
        }
        else if (grp == ATTR_POINT) {
            return m_point_attrs;
        }
        else if (grp == ATTR_FACE) {
            return m_face_attrs;
        }
        else if (grp == ATTR_VERTEX) {
            return m_vert_attrs;
        }
        else {
            throw makeError<UnimplError>("Unknown group on attr");
        }
    }

    const std::map<std::string, AttributeVector>& GeometryObject::get_const_container(GeoAttrGroup grp) const {
        if (grp == ATTR_GEO) {
            return m_geo_attrs;
        }
        else if (grp == ATTR_POINT) {
            return m_point_attrs;
        }
        else if (grp == ATTR_FACE) {
            return m_face_attrs;
        }
        else if (grp == ATTR_VERTEX) {
            return m_vert_attrs;
        }
        else {
            throw makeError<UnimplError>("Unknown group on attr");
        }
    }


    int GeometryObject::create_attr(GeoAttrGroup grp, const std::string& attr_name, const AttrVar& val_or_vec)
    {
        if (attr_name.find(' ') != std::string::npos) {
            throw makeError<UnimplError>("space is not allowed to ocurrs into the attr_name");
        }

        std::map<std::string, AttributeVector>& container = get_container(grp);
        const int n = get_attr_size(grp);
        auto iter = container.find(attr_name);
        if (iter != container.end()) {
            iter->second = AttributeVector(val_or_vec, n == 0 ? 1 : n);
            return 0;   //already exist
        }

        container.insert(std::make_pair(attr_name, AttributeVector(val_or_vec, n == 0 ? 1 : n)));
        return 0;
    }

    int GeometryObject::create_face_attr(std::string const& attr_name, const AttrVar& defl) {
        if (attr_name.find(' ') != std::string::npos) {
            throw makeError<UnimplError>("space is not allowed to ocurrs into the attr_name");
        }

        auto iter = m_face_attrs.find(attr_name);
        if (iter != m_face_attrs.end()) {
            set_face_attr(attr_name, defl);
            return -1;   //already exist
        }
        int n = m_spTopology->nfaces();
        m_face_attrs.insert(std::make_pair(attr_name, AttributeVector(defl, n)));
        return 0;
    }

    int GeometryObject::create_point_attr(std::string const& attr_name, const AttrVar& defl) {
        if (attr_name.find(' ') != std::string::npos) {
            throw makeError<UnimplError>("space is not allowed to ocurrs into the attr_name");
        }

        std::string attr = attr_name;
        if (attr == "P") {
            attr = "pos";
        }

        int n = m_spTopology->npoints();
        auto iter = m_point_attrs.find(attr);
        if (iter != m_point_attrs.end()) {
            iter->second = AttributeVector(defl, n);
            return 0;
        }
        m_point_attrs.insert(std::make_pair(attr, AttributeVector(defl, n)));
        return 0;
    }

    int GeometryObject::create_vertex_attr(std::string const& attr_name, const AttrVar& defl) {
        if (attr_name.find(' ') != std::string::npos) {
            throw makeError<UnimplError>("space is not allowed to ocurrs into the attr_name");
        }

        if (attr_name == "Point Number") {
            throw makeError<UnimplError>("Point Number is an internal attribute");
        }
        auto iter = m_vert_attrs.find(attr_name);
        if (iter != m_vert_attrs.end()) {
            return -1;
        }
        int n = m_spTopology->nvertices();
        m_vert_attrs.insert(std::make_pair(attr_name, AttributeVector(defl, n)));

        return 0;
    }

    int GeometryObject::create_geometry_attr(std::string const& attr_name, const AttrVar& defl) {
        if (attr_name.find(' ') != std::string::npos) {
            throw makeError<UnimplError>("space is not allowed to ocurrs into the attr_name");
        }

        auto iter = m_geo_attrs.find(attr_name);
        if (iter != m_geo_attrs.end()) {
            return -1;   //already exist
        }
        m_geo_attrs.insert(std::make_pair(attr_name, AttributeVector(defl, 1)));
        return 0;
    }

    void GeometryObject::copy_attr(GeoAttrGroup grp, const std::string& src_attr, const std::string& dest_attr) {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iter_source = container.find(src_attr);
        if (iter_source == container.end()) {
            throw makeError<UnimplError>("the source attr `" + src_attr + "` doesn't exist when copy attr.");
        }
        const AttributeVector& srcattr = iter_source->second;
        auto iter_dest = container.find(dest_attr);
        if (iter_dest == container.end()) {
            AttributeVector attrvec(srcattr);
            container.emplace(std::make_pair(dest_attr, std::move(attrvec)));
        }
        else {
            iter_dest->second = srcattr;
        }
    }

    void GeometryObject::copy_attr_from(GeoAttrGroup grp, GeometryObject* pSrcObject, const std::string& src_attr, const std::string& dest_attr)
    {
        std::map<std::string, AttributeVector>& src_container = pSrcObject->get_container(grp);
        std::map<std::string, AttributeVector>& dest_container = get_container(grp);

        auto iter_source = src_container.find(src_attr);
        if (iter_source == src_container.end()) {
            throw makeError<UnimplError>("the source attr `" + src_attr + "` doesn't exist when copy attr.");
        }
        const AttributeVector& srcattr = iter_source->second;

        auto iter_dest = dest_container.find(dest_attr);
        if (iter_dest == dest_container.end()) {
            AttributeVector attrvec(srcattr);
            dest_container.emplace(std::make_pair(dest_attr, std::move(attrvec)));
        }
        else {
            iter_dest->second = srcattr;
        }
    }

    int GeometryObject::delete_attr(GeoAttrGroup grp, const std::string& attr_name)
    {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iter = container.find(attr_name);
        if (iter == container.end())
            return 0;
        container.erase(iter);
        return 1;
    }

    int GeometryObject::delete_vertex_attr(std::string const& attr_name)
    {
        auto iter = m_vert_attrs.find(attr_name);
        if (iter == m_vert_attrs.end())
            return 0;
        m_vert_attrs.erase(iter);
        return 1;
    }

    int GeometryObject::delete_point_attr(std::string const& attr_name)
    {
        auto iter = m_point_attrs.find(attr_name);
        if (iter == m_point_attrs.end())
            return 0;
        m_point_attrs.erase(iter);
        return 1;
    }

    int GeometryObject::delete_face_attr(std::string const& attr_name)
    {
        auto iter = m_face_attrs.find(attr_name);
        if (iter == m_face_attrs.end())
            return 0;
        m_face_attrs.erase(iter);
        return 1;
    }

    int GeometryObject::delete_geometry_attr(std::string const& attr_name)
    {
        auto iter = m_geo_attrs.find(attr_name);
        if (iter == m_geo_attrs.end())
            return 0;
        m_geo_attrs.erase(iter);
        return 1;
    }

    AttrValue GeometryObject::get_attr_elem(GeoAttrGroup grp, const std::string& attr_name, size_t idx)
    {
        const std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iterContainer = container.find(attr_name);
        if (iterContainer != container.end()) {
            const AttributeVector& attrVec = iterContainer->second;
            return attrVec.getelem(idx);
        }
        else {
            throw makeError<UnimplError>("Unknown attr");
        }
    }

    void GeometryObject::set_attr_elem(GeoAttrGroup grp, const std::string& attr_name, size_t idx, AttrValue val)
    {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iterContainer = container.find(attr_name);
        if (iterContainer != container.end()) {
            AttributeVector& attrVec = iterContainer->second;
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (
                    std::is_same_v<T, float> ||
                    std::is_same_v<T, int> ||
                    std::is_same_v<T, vec2f> ||
                    std::is_same_v<T, vec3f> ||
                    std::is_same_v<T, vec4f> ||
                    std::is_same_v<T, std::string>
                    ) {
                    set_elem(grp, attr_name, idx, arg);
                }
            }, val);
        }
        else {
            throw makeError<UnimplError>("Unknown attr");
        }
    }

    bool GeometryObject::has_attr(GeoAttrGroup grp, std::string const& name, GeoAttrType type)
    {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto it = container.find(name);
        if (it == container.end()) return false;
        GeoAttrType attrType = it->second.type();
        if (type == ATTR_TYPE_UNKNOWN)
            return true;
        return attrType == type;
    }

    std::vector<std::string> GeometryObject::attributes(GeoAttrGroup grp) {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        std::vector<std::string> attrs;
        for (const auto& [name, _] : container) {
            attrs.push_back(name);
        }
        return attrs;
    }

    bool GeometryObject::has_vertex_attr(std::string const& name) const
    {
        return m_vert_attrs.find(name) != m_vert_attrs.end();
    }

    bool GeometryObject::has_point_attr(std::string const& name) const
    {
        return m_point_attrs.find(name) != m_point_attrs.end();
    }

    bool GeometryObject::has_face_attr(std::string const& name) const
    {
        return m_face_attrs.find(name) != m_face_attrs.end();
    }

    bool GeometryObject::has_geometry_attr(std::string const& name) const
    {
        return m_geo_attrs.find(name) != m_geo_attrs.end();
    }

#ifdef TRACE_GEOM_ATTR_DATA
    std::string GeometryObject::get_attr_data_id(GeoAttrGroup grp, std::string const& name, std::string channel) {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iter = container.find(name);
        if (iter != container.end()) {
            auto& attrVec = iter->second;
            if (channel == "") return attrVec.self_id();
            if (channel == "x") return attrVec.xcomp_id();
            if (channel == "y") return attrVec.ycomp_id();
            if (channel == "z") return attrVec.zcomp_id();
            if (channel == "w") return attrVec.wcomp_id();
        }
        return "";
    }
#endif

    GeoAttrType GeometryObject::get_attr_type(GeoAttrGroup grp, std::string const& name) {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iter = container.find(name);
        if (iter == container.end()) {
            std::string _name = name;
            if (name == "P" && grp == ATTR_POINT) {
                iter = container.find("pos");
                assert(iter != container.end());
                auto& attrVec = iter->second;
                return attrVec.type();
            }
            return ATTR_TYPE_UNKNOWN;
        }
        auto& attrVec = iter->second;
        return attrVec.type();
    }

    std::vector<std::string> GeometryObject::get_attr_names(GeoAttrGroup grp) {
        std::vector<std::string> names;
        switch (grp) {
        case ATTR_GEO:
            for (auto& [name, _] : m_geo_attrs) {
                names.push_back(name);
            }
            break;
        case ATTR_FACE:
            for (auto& [name, _] : m_face_attrs) {
                names.push_back(name);
            }
            break;
        case ATTR_POINT:
            for (auto& [name, _] : m_point_attrs) {
                names.push_back(name);
            }
            break;
        case ATTR_VERTEX:
            for (auto& [name, _] : m_vert_attrs) {
                names.push_back(name);
            }
            break;
        }
        return names;
    }

    int GeometryObject::set_attr(GeoAttrGroup grp, std::string const& name, const AttrVar& val)
    {
        std::map<std::string, AttributeVector>& container = get_container(grp);
        auto iter = container.find(name);
        if (iter == container.end()) {
            return -1;
            //throw makeError<KeyError>(name, "not exist on point attr");
        }
        AttributeVector& spAttr = iter->second;
        int n = spAttr.size();
        spAttr.set(val);
        return 0;
    }

    int GeometryObject::set_vertex_attr(std::string const& attr_name, const AttrVar& defl)
    {
        auto iter = m_vert_attrs.find(attr_name);
        if (iter == m_vert_attrs.end()) {
            return -1;
        }
        AttributeVector& spAttr = iter->second;
        int n = spAttr.size();
        spAttr.set(defl);
        return 0;
    }

    int GeometryObject::set_point_attr(std::string const& attr_name, const AttrVar& defl)
    {
        auto iter = m_point_attrs.find(attr_name);
        if (iter == m_point_attrs.end()) {
            return -1;
        }
        AttributeVector& spAttr = iter->second;
        int n = spAttr.size();
        spAttr.set(defl);
        return 0;
    }

    int GeometryObject::set_face_attr(std::string const& attr_name, const AttrVar& defl)
    {
        auto iter = m_face_attrs.find(attr_name);
        if (iter == m_face_attrs.end()) {
            return -1;
        }
        AttributeVector& spAttr = iter->second;
        int n = spAttr.size();
        spAttr.set(defl);
        return 0;
    }

    int GeometryObject::set_geometry_attr(std::string const& attr_name, const AttrVar& defl)
    {
        auto iter = m_geo_attrs.find(attr_name);
        if (iter == m_geo_attrs.end()) {
            return -1;
        }
        AttributeVector& spAttr = iter->second;
        int n = spAttr.size();
        spAttr.set(defl);
        return 0;
    }

    int GeometryObject::add_point(zeno::vec3f pos) {
        copyTopologyAccordtoUseCount();

        auto pointAttrIter = m_point_attrs.find("pos");
        if (pointAttrIter == m_point_attrs.end()) {
            create_attr(ATTR_POINT, "pos", pos);
        } else {
            pointAttrIter->second.append(pos);
        }
        int ptnum = m_spTopology->add_point();
        CALLBACK_NOTIFY(add_point, ptnum)
        return ptnum;
    }
        
    int GeometryObject::add_vertex(int face_id, int point_id) {
        copyTopologyAccordtoUseCount();
        int ret = m_spTopology->add_vertex(face_id, point_id);
        if (ret != -1) {
            CALLBACK_NOTIFY(add_vertex, m_spTopology->vertex_index(face_id, ret))
            return ret;
        } else {
            return -1;
        }
    }

    /* Vertex相关 */

    /* linear_vertex_idx 这个概念的意思是，vertex在整个几何体所有vertex上的位置索引
      比如vertex排序如下：
        0:0
        0:1
        0:2
        1:0
        1:1     ->这个vertex对应的linear_idx就是4.
        1:2
        1:3     ->这个vertex对应的linear_idx就是6.
    */

    /*
      根据face_id和vertex_id(vertex在面内部的索引，起始值为0）
      返回linear_vertex_idx.
     */
    int GeometryObject::vertex_index(int face_id, int vertex_id) {
        return m_spTopology->vertex_index(face_id, vertex_id);
    }

    /*
     * 与linear_vertex_id共享一个point的下一个vertex的linear_vertex_id;
     */
    int GeometryObject::vertex_next(int linear_vertex_id) {
        return m_spTopology->vertex_next(linear_vertex_id);
    }

    /*
     * 与linear_vertex_id共享一个point的上一个vertex的linear_vertex_id;
     */
    int GeometryObject::vertex_prev(int linear_vertex_id) {
        return m_spTopology->vertex_prev(linear_vertex_id);
    }

    /*
     * 与linear_vertex_id关联的point的id;
     */
    int GeometryObject::vertex_point(int linear_vertex_id) {
        return m_spTopology->vertex_point(linear_vertex_id);
    }

    /*
     * 与linear_vertex_id关联的face的id;
     */
    int GeometryObject::vertex_face(int linear_vertex_id) {
        return m_spTopology->vertex_face(linear_vertex_id);
    }

    /*
     * 将linear_vertex_id转为它所在的那个面上的idx（就是2:3里面的3);
     */
    int GeometryObject::vertex_face_index(int linear_vertex_id) {
        return m_spTopology->vertex_face_index(linear_vertex_id);
    }

    std::tuple<int, int, int> GeometryObject::vertex_info(int linear_vertex_id) {
        return m_spTopology->vertex_info(linear_vertex_id);
    }

    int GeometryObject::isLineFace(int faceid)
    {
        //DEPRECATED
        return 0;
    }

    int GeometryObject::add_face(const std::vector<int>& points, bool bClose) {
        copyTopologyAccordtoUseCount();
        int faceid = m_spTopology->add_face(points, bClose);
        CALLBACK_NOTIFY(add_face, faceid);
        CALLBACK_NOTIFY(reset_vertices)
        return faceid;
    }

    void GeometryObject::set_face(int idx, const std::vector<int>& points, bool bClose) {
        m_spTopology->set_face(idx, points, bClose);
    }

    /*
        从几何体中删除 face。
        andPoints = 0，不删除点。
        andPoints = 1，则还将删除与 face 关联但与任何其他 face 无关
        的任何点。
    */
    bool GeometryObject::remove_faces(const std::set<int>& faces, bool includePoints) {
        copyTopologyAccordtoUseCount();

        std::vector<int> removedPtnums;
        std::vector<int> removedVertices;
        for (auto& faceid: faces) {
            std::vector<int> vertices = m_spTopology->face_vertices(faceid);
            std::copy(vertices.begin(), vertices.end(), std::back_inserter(removedVertices));
        }

        bool ret = m_spTopology->remove_faces(faces, includePoints, removedPtnums);
        if (ret) {
            if (includePoints) {
                for (auto& [name, attrib_vec] : m_point_attrs) {
                    for (int i = removedPtnums.size() - 1; i >= 0; i--) {
                        removeAttribElem(attrib_vec, removedPtnums[i]);
                    }
                }
            }
            for (auto& [name, attrib_vec] : m_vert_attrs) {
                for (int i = removedVertices.size() - 1; i >= 0; i--) {
                    removeAttribElem(attrib_vec, removedVertices[i]);
                }
            }
            for (auto& [name, attrib_vec] : m_face_attrs) {
                for (auto it = faces.rbegin(); it != faces.rend(); ++it) {
                    removeAttribElem(attrib_vec, *it);
                }
            }
            CALLBACK_NOTIFY(reset_faces)
            CALLBACK_NOTIFY(reset_vertices)
            return true;
        } else {
            return false;
        }
    }

    int GeometryObject::npoints() const {
        return m_spTopology->npoints();
    }

    int GeometryObject::nfaces() const {
        return m_spTopology->nfaces();
    }

    int GeometryObject::nvertices() const {
        return m_spTopology->nvertices();
    }

    int GeometryObject::nvertices(int face_id) const {
        return m_spTopology->nvertices(face_id);
    }

    int GeometryObject::nattributes(GeoAttrGroup grp) const {
        switch (grp) {
        case ATTR_GEO: return m_geo_attrs.size();
        case ATTR_FACE: return m_face_attrs.size();
        case ATTR_POINT: return m_point_attrs.size();
        case ATTR_VERTEX: return m_vert_attrs.size();
        default:
            return 0;
        }
    }
}