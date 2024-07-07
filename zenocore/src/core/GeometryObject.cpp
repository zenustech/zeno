#include <zeno/types/GeometryObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <assert.h>
#include <zeno/formula/syntax_tree.h>


namespace zeno
{
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

    ZENO_API GeometryObject::GeometryObject() {

    }

    ZENO_API GeometryObject::GeometryObject(const GeometryObject& geo) {

    }

    ZENO_API GeometryObject::GeometryObject(PrimitiveObject* prim) {
        initFromPrim(prim);
    }

    ZENO_API std::shared_ptr<PrimitiveObject> GeometryObject::toPrimitive() const {
        std::shared_ptr<PrimitiveObject> spPrim = std::make_shared<PrimitiveObject>();
        assert(m_points.size() == m_points_data.size());
        spPrim->verts = m_points_data;

        int startIdx = 0;
        if (m_bTriangle) {
            spPrim->tris->resize(m_faces.size());
        }
        else {
            spPrim->polys->resize(m_faces.size());
        }

        for (int i = 0; i < m_faces.size(); i++) {
            auto& face = m_faces[i];
            int firsth = face.h, h = firsth;
            std::vector<int> points;
            do {
                const auto& hedge = m_hEdges[h];
                points.push_back(hedge.point);
                h = hedge.next;
            } while (firsth != h);

            if (m_bTriangle) {
                vec3i tri = { points[0],points[1], points[2] };
                spPrim->tris[i] = std::move(tri);
            }
            else {
                int sz = points.size();
                for (auto pt : points) {
                    spPrim->loops.push_back(pt);
                }
                spPrim->polys.push_back({startIdx, sz});
                startIdx += sz;
            }
        }
        return spPrim;
    }

    int GeometryObject::checkHEdge(int fromPoint, int toPoint) {
        assert(fromPoint < m_points.size());
        for (auto hedge : m_points[fromPoint].edges) {
            if (m_hEdges[hedge].point == toPoint) {
                return hedge;
            }
        }
        return -1;
    }

    int GeometryObject::getNextOutEdge(int fromPoint, int currentOutEdge) {
        return -1;
    }

    bool GeometryObject::has_point_attr(std::string const& name) const {
        return m_points_data.has_attr(name);
    }

    void GeometryObject::initFromPrim(PrimitiveObject* prim) {

        m_points.resize(prim->verts->size());
        m_points_data = prim->verts;

        int nFace = -1;
        m_bTriangle = prim->loops->empty() && !prim->tris->empty();
        if (m_bTriangle) {
            nFace = prim->tris->size();
            m_hEdges.reserve(nFace * 3);
        }
        else {
            assert(!prim->loops->empty() && !prim->polys->empty());
            nFace = prim->polys->size();
            //一般是四边形
            m_hEdges.reserve(nFace * 4);
        }
        m_faces.resize(nFace);

        for (int face = 0; face < nFace; face++) {
            std::vector<int> points;
            if (m_bTriangle) {
                auto const& ind = prim->tris[face];
                points = { ind[0], ind[1], ind[2] };
            }
            else {
                auto const& poly = prim->polys[face];
                int startIdx = poly[0], nPoints = poly[1];
                auto const& loops = prim->loops;
                for (int i = 0; i < nPoints; i++) {
                    points.push_back(loops[startIdx + i]);
                }
            }

            Face f;
            int lastHedge = -1, firstHedge = -1;
            for (int i = 0; i < points.size(); i++) {
                int vp = -1, vq = -1;
                if (i < points.size() - 1) {
                    vp = points[i];
                    vq = points[i + 1];
                }
                else {
                    vp = points[i];
                    vq = points[0];
                }

                //vp->vq
                int hpq = -1;

                HEdge hedge;
                hedge.face = face;
                hedge.point = vq;
                hpq = m_hEdges.size();

                if (lastHedge != -1) {
                    m_hEdges[lastHedge].next = hpq;
                }
                //TODO: 如果只有一条边会怎么样？
                if (i == points.size() - 1) {
                    hedge.next = firstHedge;
                }
                else if (i == 0) {
                    firstHedge = hpq;
                }

                //check whether the pair edge exist
                int pairedge = checkHEdge(vq, vp);
                if (pairedge >= 0) {
                    hedge.pair = pairedge;
                    m_hEdges[pairedge].pair = hpq;
                }

                m_hEdges.emplace_back(hedge);
                m_points[vp].edges.insert(hpq);

                f.h = hpq;

                lastHedge = hpq;
            }
            m_faces[face] = f;
        }
    }

    int GeometryObject::get_point_count() const {
        return m_points.size();
    }

    int GeometryObject::get_face_count() const {
        return m_faces.size();
    }

    std::vector<vec3f> GeometryObject::get_points() const {
        return m_points_data;
    }

    void GeometryObject::set_points_pos(const ZfxVariable& val, ZfxElemFilter& filter) {
        for (int i = 0; i < m_points.size(); i++) {
            if (filter[i]) {
                const glm::vec3& vec = get_zfxvar<glm::vec3>(val.value[i]);
                m_points_data[i] = { vec.x, vec.y, vec.z };
            }
        }
    }

    void GeometryObject::set_points_normal(const ZfxVariable& val, ZfxElemFilter& filter)
    {
        std::vector<vec3f>& nrms = m_points_data.attr<vec3f>("nrm");
        for (int i = 0; i < m_points_data.size(); i++) {
            if (filter[i]) {
                const glm::vec3& vec = get_zfxvar<glm::vec3>(val.value[i]);
                nrms[i] = { vec.x, vec.y, vec.z };
            }
        }
    }

    int get_adjust_position(const std::vector<int>& remIndice, int oldindex) {
        int numOfExceed = 0;
        for (int remIdx : remIndice) {
            if (oldindex >= remIdx)
                numOfExceed++;
            else
                break;
        }
        return oldindex - numOfExceed;
    }

    bool GeometryObject::remove_point(int ptnum) {
        if (m_bTriangle) {
            std::set<int> _remFaces, _remHEdges;
            if (ptnum < 0 || ptnum >= m_points.size())
                return false;

            for (int outEdge : m_points[ptnum].edges) {
                assert(outEdge >= 0 && outEdge < m_hEdges.size());
                auto& hedge = m_hEdges[outEdge];
                _remFaces.insert(hedge.face);

                int h = outEdge;
                do {
                    _remHEdges.insert(h);
                    hedge = m_hEdges[h];
                    h = hedge.next;
                } while (h != outEdge);
            }

            std::vector<int> remFaces, remHEdges;
            for (int idx : _remFaces)
                remFaces.push_back(idx);
            for (int idx : _remHEdges)
                remHEdges.push_back(idx);

            std::sort(remFaces.begin(), remFaces.end());
            std::sort(remHEdges.begin(), remHEdges.end());

            std::map<int, int> edgeIdxMapper;
            for (int i = 0; i < m_hEdges.size(); i++) {
                if (_remHEdges.find(i) != _remHEdges.end())
                    continue;
                int newIdx = get_adjust_position(remHEdges, i);
                edgeIdxMapper.insert(std::make_pair(i, newIdx));
            }

            std::map<int, int> faceIdxMapper;
            for (int i = 0; i < m_faces.size(); i++) {
                if (_remFaces.find(i) != _remFaces.end())
                    continue;
                int newIdx = get_adjust_position(remFaces, i);
                faceIdxMapper.insert(std::make_pair(i, newIdx));
            }

            m_points.erase(m_points.begin() + ptnum);
            m_points_data.values.erase(m_points_data.values.begin() + ptnum);
            for (auto iter = m_points_data.attrs.begin(); iter != m_points_data.attrs.end(); iter++)
            {
                std::visit([&](auto& val) {
                    val.erase(val.begin() + ptnum);
                }, iter->second);
            }

            //adjust all points
            for (int i = 0; i < m_points.size(); i++) {
                std::set<int> edges;
                for (int h : m_points[i].edges) {
                    edges.insert(edgeIdxMapper[h]);
                }
                m_points[i].edges = edges;
            }

            //adjust edges
            for (int i = 0; i < m_hEdges.size(); i++)
            {
                auto& hedge = m_hEdges[i];
                if (_remHEdges.find(i) != _remHEdges.end()) {
                    hedge.face = -1;
                }
                else {
                    hedge.pair = edgeIdxMapper[hedge.pair];
                    hedge.next = edgeIdxMapper[hedge.next];
                    hedge.face = faceIdxMapper[hedge.face];
                    if (hedge.point >= ptnum) {
                        hedge.point--;
                    }
                }
            }
            for (auto iter = m_hEdges.begin(); iter != m_hEdges.end();) {
                if (iter->face == -1) {
                    iter = m_hEdges.erase(iter);
                }
                else {
                    iter++;
                }
            }

            //adjust face
            for (int i = m_faces.size() - 1; i >= 0; i--) {
                if (_remFaces.find(i) != _remFaces.end()) {
                    m_faces.erase(m_faces.begin() + i);
                }
                else {
                    m_faces[i].h = edgeIdxMapper[m_faces[i].h];
                }
            }
        }
    }
}