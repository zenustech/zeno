#include <zeno/types/GeometryObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/helper.h>
#include <zeno/utils/interfaceutil.h>
#include <zeno/core/ObjectRecorder.h>


namespace zeno
{
//#define TEST_GEOMETRY_LEAK

    zany GeometryObject_Adapter::clone() const {
        auto newGeom = std::make_unique<GeometryObject_Adapter>();
        newGeom->m_impl = std::make_unique<GeometryObject>(*m_impl);
        newGeom->m_usrData = this->m_usrData->clone();  //TODO:调整写法
        newGeom->m_key = this->m_key;
        return newGeom;
    }

    void GeometryObject_Adapter::Delete() {}

    GeometryObject_Adapter::GeometryObject_Adapter() {
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geoms.insert(this);
#endif
    }

    GeometryObject_Adapter::~GeometryObject_Adapter() {
        Delete();
#ifdef TEST_GEOMETRY_LEAK
        zeno::getSession().m_recorder->m_geoms.erase(this);
#endif
    }

    void GeometryObject_Adapter::inheritAttributes(
        GeometryObject_Adapter* rhs,
        int vtx_offset,
        int pt_offset,
        std::set<zeno::String> pt_nocopy,    //TODO abi problem
        int face_offset,
        std::set<zeno::String> face_nocopy
    ) {
        std::set<std::string> std_pt_nocopy, std_face_nocopy;
        for (auto zs : pt_nocopy) {
            std_pt_nocopy.insert(zeno::zsString2Std(zs));
        }
        for (auto zs : face_nocopy) {
            std_face_nocopy.insert(zeno::zsString2Std(zs));
        }
        m_impl->inheritAttributes(rhs->m_impl.get(), vtx_offset, pt_offset, std_pt_nocopy, face_offset, std_face_nocopy);
    }

    std::vector<zeno::vec3f> GeometryObject_Adapter::points_pos() {
        return m_impl->points_pos();
    }

    Vector<Vec3i> GeometryObject_Adapter::tri_indice() const {
        const std::vector<vec3f>& vec = m_impl->points_pos();
        Vector<Vec3i> _vec(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            _vec[i] = toAbiVec3i(vec[i]);
        }
        return _vec;
    }

    Vector<int> GeometryObject_Adapter::edge_list() const {
        return stdVec2zeVec(m_impl->edge_list());
    }

    void GeometryObject_Adapter::set_pos(int i, zeno::vec3f pos) {
        m_impl->set_attr_elem(ATTR_POINT, "pos", i, pos);
    }

    bool GeometryObject_Adapter::is_base_triangle() const {
        return m_impl->is_base_triangle();
    }

    bool GeometryObject_Adapter::is_Line() const {
        return m_impl->is_Line();
    }

    int GeometryObject_Adapter::get_group_count(GeoAttrGroup grp) const {
        return m_impl->get_group_count(grp);
    }

    GeoAttrType GeometryObject_Adapter::get_attr_type(GeoAttrGroup grp, const zeno::String& name) {
        return m_impl->get_attr_type(grp, zsString2Std(name));
    }

    zeno::Vector<zeno::String> GeometryObject_Adapter::get_attr_names(GeoAttrGroup grp) {
        const std::vector<std::string>& vec = m_impl->get_attr_names(grp);
        Vector<zeno::String> _vec(vec.size());
        for (int i = 0; i < vec.size(); i++) {
            _vec[i] = stdString2zs(vec[i]);
        }
        return _vec;
    }

    //创建属性
    int GeometryObject_Adapter::create_attr(GeoAttrGroup grp, const zeno::String& attr_name, const Any& defl) {
        return m_impl->create_attr(grp, zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::create_face_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->create_face_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::create_point_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->create_point_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::create_vertex_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->create_vertex_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::create_geometry_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->create_geometry_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    std::vector<std::string> GeometryObject_Adapter::attributes(GeoAttrGroup grp) {
        return m_impl->attributes(grp);
    }

    void GeometryObject_Adapter::copy_attr(GeoAttrGroup grp, const zeno::String& src_attr, const zeno::String& dest_attr) {
        m_impl->copy_attr(grp, zsString2Std(src_attr), zsString2Std(dest_attr));
    }

    void GeometryObject_Adapter::copy_attr_from(GeoAttrGroup grp, GeometryObject_Adapter* pSrcObject, const zeno::String& src_attr, const zeno::String& dest_attr) {
        m_impl->copy_attr_from(grp, pSrcObject->m_impl.get(), zsString2Std(src_attr), zsString2Std(dest_attr));
    }

    std::vector<float> GeometryObject_Adapter::get_float_attr(GeoAttrGroup grp, const zeno::String& attr_name) {
        return m_impl->get_attrs<float>(grp, zsString2Std(attr_name));
    }

    std::vector<vec3f> GeometryObject_Adapter::get_vec3f_attr(GeoAttrGroup grp, const zeno::String& attr_name) {
        return m_impl->get_attrs<vec3f>(grp, zsString2Std(attr_name));
    }

    int GeometryObject_Adapter::set_attr(GeoAttrGroup grp, const zeno::String& name, const Any& val) {
        return m_impl->set_attr(grp, zsString2Std(name), abiAnyToAttrVar(val));
    }

    int GeometryObject_Adapter::set_vertex_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->set_vertex_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::set_point_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->set_point_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::set_face_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->set_face_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }

    int GeometryObject_Adapter::set_geometry_attr(const zeno::String& attr_name, const Any& defl) {
        return m_impl->set_geometry_attr(zsString2Std(attr_name), abiAnyToAttrVar(defl));
    }


    void GeometryObject_Adapter::foreach_vec3_attr_update(
        GeoAttrGroup grp,
        const zeno::String& attr_name,
        char channel,
        std::function<zeno::vec3f(int idx, zeno::vec3f old_elem_value)>&& evalf) {
        m_impl->foreach_attr_update<zeno::vec3f>(grp, zsString2Std(attr_name), channel, std::move(evalf));
    }

    void GeometryObject_Adapter::foreach_float_attr_update(
        GeoAttrGroup grp,
        const zeno::String& attr_name,
        char channel,
        std::function<float(int idx, float old_elem_value)>&& evalf) {
        m_impl->foreach_attr_update<float>(grp, zsString2Std(attr_name), channel, std::move(evalf));
    }

    bool GeometryObject_Adapter::has_attr(GeoAttrGroup grp, const zeno::String& name, GeoAttrType type) {
        return m_impl->has_attr(grp, zsString2Std(name));
    }

    bool GeometryObject_Adapter::has_vertex_attr(const zeno::String& name) const {
        return m_impl->has_vertex_attr(zsString2Std(name));
    }

    bool GeometryObject_Adapter::has_point_attr(const zeno::String& name) const {
        return m_impl->has_point_attr(zsString2Std(name));
    }

    bool GeometryObject_Adapter::has_face_attr(const zeno::String& name) const {
        return m_impl->has_face_attr(zsString2Std(name));
    }

    bool GeometryObject_Adapter::has_geometry_attr(const zeno::String& name) const {
        return m_impl->has_geometry_attr(zsString2Std(name));
    }

    int GeometryObject_Adapter::delete_attr(GeoAttrGroup grp, const zeno::String& attr_name) {
        return m_impl->delete_attr(grp, zsString2Std(attr_name));
    }

    int GeometryObject_Adapter::delete_vertex_attr(const zeno::String& attr_name) {
        return m_impl->delete_vertex_attr(zsString2Std(attr_name));
    }

    int GeometryObject_Adapter::delete_point_attr(const zeno::String& attr_name) {
        return m_impl->delete_point_attr(zsString2Std(attr_name));
    }

    int GeometryObject_Adapter::delete_face_attr(const zeno::String& attr_name) {
        return m_impl->delete_face_attr(zsString2Std(attr_name));
    }

    int GeometryObject_Adapter::delete_geometry_attr(const zeno::String& attr_name) {
        return m_impl->delete_geometry_attr(zsString2Std(attr_name));
    }

    int GeometryObject_Adapter::add_vertex(int face_id, int point_id) {
        return m_impl->add_vertex(face_id, point_id);
    }

    int GeometryObject_Adapter::add_point(Vec3f pos) {
        return m_impl->add_point(toVec3f(pos));
    }

    int GeometryObject_Adapter::add_face(const zeno::Vector<int>& points, bool bClose) {
        return m_impl->add_face(zeVec2stdVec(points), bClose);
    }

    void GeometryObject_Adapter::set_face(int idx, const zeno::Vector<int>& points, bool bClose) {
        return m_impl->set_face(idx, zeVec2stdVec(points), bClose);
    }

    bool GeometryObject_Adapter::remove_faces(const std::set<int>& faces, bool includePoints) {
        return m_impl->remove_faces(faces, includePoints);
    }

    bool GeometryObject_Adapter::remove_point(int ptnum) {
        return m_impl->remove_point(ptnum);
    }

    bool GeometryObject_Adapter::remove_vertex(int face_id, int vert_id) {
        return m_impl->remove_vertex(face_id, vert_id);
    }

    int GeometryObject_Adapter::npoints() const {
        return m_impl->npoints();
    }

    int GeometryObject_Adapter::nfaces() const {
        return m_impl->nfaces();
    }

    int GeometryObject_Adapter::nvertices() const {
        return m_impl->nvertices();
    }

    int GeometryObject_Adapter::nvertices(int face_id) const {
        return m_impl->nvertices(face_id);
    }

    int GeometryObject_Adapter::nattributes(GeoAttrGroup grp) const {
        return m_impl->nattributes(grp);
    }

    zeno::Vector<int> GeometryObject_Adapter::point_faces(int point_id) {
        return stdVec2zeVec(m_impl->point_faces(point_id));
    }

    int GeometryObject_Adapter::point_vertex(int point_id) {
        return m_impl->point_vertex(point_id);
    }

    zeno::Vector<int> GeometryObject_Adapter::point_vertices(int point_id) {
        return stdVec2zeVec(m_impl->point_vertices(point_id));
    }

    int GeometryObject_Adapter::face_point(int face_id, int vert_id) const {
        return m_impl->face_point(face_id, vert_id);
    }

    zeno::Vector<int> GeometryObject_Adapter::face_points(int face_id) {
        return stdVec2zeVec(m_impl->face_points(face_id));
    }

    int GeometryObject_Adapter::face_vertex(int face_id, int vert_id) {
        return m_impl->face_vertex(face_id, vert_id);
    }

    int GeometryObject_Adapter::face_vertex_count(int face_id) {
        return m_impl->face_vertex_count(face_id);
    }

    zeno::Vector<int> GeometryObject_Adapter::face_vertices(int face_id) {
        return stdVec2zeVec(m_impl->face_vertices(face_id));
    }

    zeno::Vec3f GeometryObject_Adapter::face_normal(int face_id) {
        return toAbiVec3f(m_impl->face_normal(face_id));
    }

    int GeometryObject_Adapter::vertex_index(int face_id, int vertex_id) {
        return m_impl->vertex_index(face_id, vertex_id);
    }

    int GeometryObject_Adapter::vertex_next(int linear_vertex_id) {
        return m_impl->vertex_next(linear_vertex_id);
    }

    int GeometryObject_Adapter::vertex_prev(int linear_vertex_id) {
        return m_impl->vertex_prev(linear_vertex_id);
    }

    int GeometryObject_Adapter::vertex_point(int linear_vertex_id) {
        return m_impl->vertex_point(linear_vertex_id);
    }

    int GeometryObject_Adapter::vertex_face(int linear_vertex_id) {
        return m_impl->vertex_face(linear_vertex_id);
    }

    int GeometryObject_Adapter::vertex_face_index(int linear_vertex_id) {
        return m_impl->vertex_face_index(linear_vertex_id);
    }

    std::tuple<int, int, int> GeometryObject_Adapter::vertex_info(int linear_vertex_id) {
        return m_impl->vertex_info(linear_vertex_id);
    }

    std::unique_ptr<PrimitiveObject> GeometryObject_Adapter::toPrimitiveObject() const {
        std::unique_ptr<PrimitiveObject> spPrim = m_impl->toPrimitive();
        spPrim->m_usrData = m_usrData->clone();
        return spPrim;
    }

    std::unique_ptr<GeometryObject_Adapter> GeometryObject_Adapter::toIndiceMeshesTopo() const {
        auto newGeom = std::make_unique<GeometryObject_Adapter>();
        newGeom->m_impl = m_impl->toIndiceMeshesTopo();
        newGeom->m_usrData = this->m_usrData->clone();
        return newGeom;
    }

    std::unique_ptr<GeometryObject_Adapter> GeometryObject_Adapter::toHalfEdgeTopo() const {
        auto newGeom = std::make_unique<GeometryObject_Adapter>();
        newGeom->m_impl = m_impl->toHalfEdgeTopo();
        newGeom->m_usrData = this->m_usrData->clone();
        return newGeom;
    }

    std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(GeomTopoType type) {
        auto pGeom = std::make_unique<GeometryObject_Adapter>();
        pGeom->m_impl = std::make_unique<GeometryObject>(type);
        return pGeom;
    }

    std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(GeomTopoType type, bool bTriangle, int nPoints, int nFaces, bool bInitFaces) {
        auto pGeom = std::make_unique<GeometryObject_Adapter>();
        pGeom->m_impl = std::make_unique<GeometryObject>(type, bTriangle, nPoints, nFaces, bInitFaces);
        //提前添加pos
        std::vector<vec3f> points(nPoints);
        pGeom->create_attr(ATTR_POINT, "pos", points);
        return pGeom;
    }

    std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(
        GeomTopoType type,
        bool bTriangle,
        const std::vector<zeno::vec3f>& points,
        const std::vector<std::vector<int>>& faces)
    {
        auto pGeom = std::make_unique<GeometryObject_Adapter>();
        pGeom->m_impl = std::make_unique<GeometryObject>(type, bTriangle, (int)points.size(), faces);
        pGeom->m_impl->create_attr(ATTR_POINT, "pos", points);
        return pGeom;
    }

    std::unique_ptr<GeometryObject_Adapter> create_GeometryObject(PrimitiveObject* prim) {
        if (!prim || prim->verts->empty())
            return nullptr;
        auto pGeom = std::make_unique<GeometryObject_Adapter>();
        pGeom->m_impl = std::make_unique<GeometryObject>(prim);
        pGeom->m_usrData = prim->m_usrData->clone();
        return pGeom;
    }

    std::unique_ptr<GeometryObject_Adapter> clone_GeometryObject(std::unique_ptr<GeometryObject_Adapter> pGeom) {
        auto newGeom = std::make_unique<GeometryObject_Adapter>();
        newGeom->m_impl = std::make_unique<GeometryObject>(*pGeom->m_impl);
        newGeom->m_usrData = pGeom->m_usrData->clone();
        return newGeom;
    }

}