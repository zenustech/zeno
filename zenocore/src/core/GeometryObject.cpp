#include <zeno/types/GeometryObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <assert.h>
#include <zeno/formula/syntax_tree.h>
#include <zeno/utils/vectorutil.h>
#include <zeno/utils/format.h>


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

    ZENO_API GeometryObject::GeometryObject(const GeometryObject& rhs) {
        m_bTriangle = rhs.m_bTriangle;
#if 0
        auto spPrim = rhs.toPrimitive();
        initFromPrim(spPrim.get());
#else
        m_points.resize(rhs.m_points.size());
        m_faces.resize(rhs.m_faces.size());

        for (int i = 0; i < m_points.size(); i++) {
            auto p = std::make_shared<Point>();
            auto& rp = rhs.m_points[i];
            p->pos = rp->pos;
            p->attr = rp->attr;
            p->normal = rp->normal;
            m_points[i] = std::move(p);
        }

        for (auto&[keyName, rEdge] : rhs.m_hEdges) {
            auto spEdge = std::make_shared<HEdge>();
            spEdge->id = rEdge->id;
            spEdge->point = rEdge->point;
            spEdge->face = rEdge->face;
            m_hEdges.insert(std::make_pair(spEdge->id, spEdge));
        }

        for (int i = 0; i < m_faces.size(); i++) {
            m_faces[i] = std::make_shared<Face>();
        }

        //adjust all pointers
        for (int i = 0; i < m_points.size(); i++) {
            for (auto rEdge : rhs.m_points[i]->edges) {
                auto& spEdge = m_hEdges[rEdge->id];
                m_points[i]->edges.insert(spEdge.get());
            }
        }

        for (auto& [keyName, rhsptr] : rhs.m_hEdges)
        {
            auto p = m_hEdges[keyName];
            p->face = rhsptr->face;
            p->point = rhsptr->point;
            if (rhsptr->pair)
                p->pair = m_hEdges[rhsptr->pair->id].get();
            p->next = m_hEdges[rhsptr->next->id].get();
        }

        for (int i = 0; i < m_faces.size(); i++) {
            auto p = m_faces[i].get();
            auto spEdge = m_hEdges[rhs.m_faces[i]->h->id];
            m_faces[i]->h = spEdge.get();
        }
#endif
    }

    ZENO_API GeometryObject::~GeometryObject() {
    }

    ZENO_API GeometryObject::GeometryObject(PrimitiveObject* prim) {
        initFromPrim(prim);
    }

    ZENO_API std::shared_ptr<PrimitiveObject> GeometryObject::toPrimitive() const {
        std::shared_ptr<PrimitiveObject> spPrim = std::make_shared<PrimitiveObject>();
        spPrim->resize(m_points.size());
        for (int i = 0; i < m_points.size(); i++) {
            spPrim->verts[i] = m_points[i]->pos;
        }

        int startIdx = 0;
        if (m_bTriangle) {
            spPrim->tris->resize(m_faces.size());
        }
        else {
            spPrim->polys->resize(m_faces.size());
        }

        for (int i = 0; i < m_faces.size(); i++) {
            auto face = m_faces[i].get();
            HEdge* firsth = face->h;
            HEdge* h = firsth;
            std::vector<int> points;
            do {
                int index = getPointTo(h);
                points.push_back(index);
                h = h->next;
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

    HEdge* GeometryObject::checkHEdge(int fromPoint, int toPoint) {
        assert(fromPoint < m_points.size() && toPoint < m_points.size() &&
               fromPoint >= 0 && toPoint >= 0);
        for (auto hedge : m_points[fromPoint]->edges) {
            if (hedge->point == toPoint) {
                return hedge;
            }
        }
        return nullptr;
    }

    int GeometryObject::getNextOutEdge(int fromPoint, int currentOutEdge) {
        return -1;
    }

    int GeometryObject::getPointTo(HEdge* hedge) const {
        return hedge->point;
    }

    bool GeometryObject::has_point_attr(std::string const& name) const {
        if (m_points.empty())
            return false;

        if (name == "pos" || name == "nrm")
            return true;

        auto attr = m_points[0]->attr;
        return attr.find(name) != attr.end();
    }

    void GeometryObject::initFromPrim(PrimitiveObject* prim) {

        m_points.resize(prim->verts->size());
        for (int i = 0; i < m_points.size(); i++) {
            m_points[i] = std::make_shared<Point>();
            m_points[i]->pos = prim->verts[i];
        }

        int nFace = -1;
        m_bTriangle = prim->loops->empty() && !prim->tris->empty();
        if (m_bTriangle) {
            nFace = prim->tris->size();
            m_hEdges.reserve(nFace * 3);
        }
        else {
            assert(!prim->loops->empty() && !prim->polys->empty());
            nFace = prim->polys->size();
            //һ�����ı���
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

            //TODO: init data for Face
            auto pFace = std::make_shared<Face>();

            HEdge* lastHedge = 0, *firstHedge = 0;
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
                auto hedge = std::make_shared<HEdge>();
                std::string id = zeno::format("{}->{}", vp, vq);
                hedge->id = id;
                m_hEdges.insert(std::make_pair(id, hedge));

                hedge->face = face;
                hedge->point = vq;

                if (lastHedge) {
                    lastHedge->next = hedge.get();
                }
                //TODO: ���ֻ��һ���߻���ô����
                if (i == points.size() - 1) {
                    hedge->next = firstHedge;
                }
                else if (i == 0) {
                    firstHedge = hedge.get();
                }

                //check whether the pair edge exist
                auto pairedge = checkHEdge(vq, vp);
                if (pairedge) {
                    hedge->pair = pairedge;
                    pairedge->pair = hedge.get();
                }

                m_points[vp]->edges.insert(hedge.get());

                //pFace->h = hedge.get();
                lastHedge = hedge.get();
            }
            //ͳһȡ��һ�������Ϊface�ġ����ߡ�
            pFace->h = firstHedge;

            m_faces[face] = std::move(pFace);
        }
    }

    int GeometryObject::get_point_count() const {
        return m_points.size();
    }

    int GeometryObject::get_face_count() const {
        return m_faces.size();
    }

    std::vector<vec3f> GeometryObject::get_points() const {
        std::vector<vec3f> pos;
        pos.resize(m_points.size());
        for (int i = 0; i < m_points.size(); i++) {
            pos[i] = m_points[i]->pos;
        }
        return pos;
    }

    std::vector<zfxvariant> GeometryObject::get_point_attr(std::string const& name) const {
        std::vector<zfxvariant> res;
        if (!has_point_attr(name))
            return res;
        res.resize(m_points.size());
        bool bPos = name == "pos";
        bool bNrm = name == "nrm";
        for (int i = 0; i < m_points.size(); i++) {
            auto pt = m_points[i].get();
            if (bPos) {
                res.push_back(glm::vec3(pt->pos[0], pt->pos[1], pt->pos[2]));
            }
            else if (bNrm) {
                res.push_back(glm::vec3(pt->normal[0], pt->normal[1], pt->normal[2]));
            }
            else {
                auto iter = pt->attr.find(name);
                assert(iter != pt->attr.end());
                res[i] = iter->second;
            }
        }
        return res;
    }

    void GeometryObject::set_points_pos(const ZfxVariable& val, ZfxElemFilter& filter) {
        for (int i = 0; i < m_points.size(); i++) {
            if (filter[i]) {
                const glm::vec3& vec = get_zfxvar<glm::vec3>(val.value[i]);
                m_points[i]->pos = { vec.x, vec.y, vec.z };
            }
        }
    }

    void GeometryObject::set_points_normal(const ZfxVariable& val, ZfxElemFilter& filter)
    {
        for (int i = 0; i < m_points.size(); i++) {
            if (filter[i]) {
                const glm::vec3& vec = get_zfxvar<glm::vec3>(val.value[i]);
                m_points[i]->normal = { vec.x, vec.y, vec.z };
            }
        }
    }

    std::tuple<Point*, HEdge*, HEdge*> GeometryObject::getPrev(HEdge* outEdge) {
        HEdge* h = outEdge, *prev = nullptr;
        Point* point = nullptr;
        do {
            prev = h;
            point = m_points[h->point].get();
            h = h->next;
        } while (h && h->next != outEdge);
        return { point, h, prev };
    }

    bool GeometryObject::remove_point(int ptnum) {
        if (ptnum < 0 || ptnum >= m_points.size())
            return false;

        std::set<int> remFaces;
        std::set<std::string> remHEdges;

        for (auto outEdge : m_points[ptnum]->edges) {
            assert(outEdge);

            HEdge* firstEdge = outEdge;
            HEdge* nextEdge = firstEdge->next;
            assert(nextEdge);
            HEdge* nnextEdge = nextEdge->next;
            assert(nnextEdge);

            auto& [prevPoint, prevEdge, pprevEdge]= getPrev(outEdge);
            assert(prevEdge && pprevEdge);
            if (nextEdge && nnextEdge == prevEdge) {
                //triangle����������������������İ�߶�Ҫ�Ƴ�
                remFaces.insert(outEdge->face);

                HEdge* h = outEdge;
                HEdge* prev = nullptr;
                do {
                    remHEdges.insert(h->id);
                    //�������ÿ��Լ�
                    if (h->pair)
                        h->pair->pair = nullptr;
                    if (prev) {
                        //��ǰ�ߵ�����edgesҲ��Ҫ����Լ�����һ���ߵĶ�����⣨��������Ҫ��ɾ���ĵ㣩
                        m_points[prev->point]->edges.erase(h);
                    }
                    prev = h;
                    h = h->next;
                } while (h != outEdge);
            }
            else {
                remHEdges.insert(outEdge->id);
                remHEdges.insert(prevEdge->id);

                auto newEdge = std::make_shared<HEdge>();

                std::string id = generateUUID();
                newEdge->id = id;
                m_hEdges.insert(std::make_pair(id, newEdge));

                //connect between outEdge->point and prevPoint.
                newEdge->point = outEdge->point;
                newEdge->pair = nullptr;
                newEdge->next = nextEdge;
                newEdge->face = outEdge->face;

                m_faces[newEdge->face]->h = newEdge.get();

                pprevEdge->next = newEdge.get();
                prevPoint->edges.erase(prevEdge);
                prevPoint->edges.insert(newEdge.get());
            }
        }

        m_points.erase(m_points.begin() + ptnum);

        for (auto keyname : remHEdges) {
            m_hEdges.erase(keyname);
        }

        //adjust face
        std::vector<int> _remFaces;
        for (int idx : remFaces)
            _remFaces.push_back(idx);
        std::sort(_remFaces.begin(), _remFaces.end());

        for (auto iter = _remFaces.rbegin(); iter != _remFaces.rend(); iter++) {
            int rmIdx = *iter;
            m_faces.erase(m_faces.begin() + rmIdx);
        }

        for (auto& [_, hedge] : m_hEdges) {
            if (hedge->point >= ptnum) {
                hedge->point--;
            }
            int nStep = 0;
            for (auto remFaceId : _remFaces) {
                if (hedge->face >= remFaceId)
                    nStep++;
                else
                    break;
            }
            hedge->face -= nStep;
            /*
            auto ph = hedge.get();
            auto fh = m_faces[hedge->face]->h;
            if (m_bTriangle) {
                assert(fh == ph || fh == ph->next || fh == ph->next->next);
            }
            */
        }
    }

    int GeometryObject::facepoint(int face_id, int vert_id) const
    {
        if (face_id < 0 || face_id >= m_faces.size())
            return -1;

        auto h = m_faces[face_id]->h;
        auto firsth = h;
        do {
            if (vert_id-- == 0) {
                return h->point;
            }
            h = h->next;
        } while (h != firsth);
        return -1;
    }

    zfxintarr GeometryObject::facepoints(int face_id)
    {
        zfxintarr pts;
        if (face_id < 0 || face_id >= m_faces.size())
            return pts;

        auto h = m_faces[face_id]->h;
        auto firsth = h;
        do {
            pts.push_back(h->point);
            h = h->next;
        } while (h != firsth);
        return pts;
    }

    zfxintarr GeometryObject::pointfaces(int point_id)
    {
        zfxintarr faces;
        if (point_id < 0 || point_id >= m_points.size())
            return faces;

        for (auto h : m_points[point_id]->edges) {
            faces.push_back(h->face);
        }
        return faces;
    }

    zfxintarr GeometryObject::pointvertex(int point_id)
    {
        throw makeError<UnimplError>();
    }

    bool GeometryObject::createFaceAttr(const std::string& attr_name, const zfxvariant& defl)
    {
        if (m_faces.empty())
            return false;

        if (m_faces[0]->attr.find(attr_name) != m_faces[0]->attr.end())
            return false;   //already exist

        for (auto pFace : m_faces) {
            pFace->attr.insert(std::make_pair(attr_name, defl));
        }
        //��Ҫͬ��������zfx manager
        return true;
    }

    bool GeometryObject::setFaceAttr(const std::string& attr_name, const zfxvariant& val)
    {
        if (m_faces.empty())
            return false;
        if (m_faces[0]->attr.find(attr_name) == m_faces[0]->attr.end())
            return false;   //not exist
        for (auto pFace : m_faces) {
            pFace->attr[attr_name] = val;
        }
        return true;
    }

    std::vector<zfxvariant> GeometryObject::getFaceAttr(const std::string& attr_name) const
    {
        std::vector<zfxvariant> res;
        res.resize(m_faces.size());
        for (int i = 0; i < m_faces.size(); i++)
        {
            assert(m_faces[i] && m_faces[i]->attr.find(attr_name) != m_faces[i]->attr.end());
            res[i] = m_faces[i]->attr[attr_name];
        }
        return res;
    }

    bool GeometryObject::deleteFaceAttr(const std::string& attr_name)
    {
        for (int i = 0; i < m_faces.size(); i++)
        {
            assert(m_faces[i] && m_faces[i]->attr.find(attr_name) != m_faces[i]->attr.end());
            m_faces[i]->attr.erase(attr_name);
        }
        return false;
    }

    bool GeometryObject::createPointAttr(const std::string& attr_name, const zfxvariant& defl) {
        if (m_points.empty())
            return false;

        if (m_points[0]->attr.find(attr_name) != m_points[0]->attr.end())
            return false;   //already exist

        for (auto pFace : m_points) {
            pFace->attr.insert(std::make_pair(attr_name, defl));
        }
        //��Ҫͬ��������zfx manager
        return true;
    }

    bool GeometryObject::setPointAttr(const std::string& attr_name, const zfxvariant& val) {
        if (m_points.empty())
            return false;
        if (m_points[0]->attr.find(attr_name) == m_points[0]->attr.end())
            return false;   //not exist
        for (auto pFace : m_points) {
            pFace->attr[attr_name] = val;
        }
        return true;
    }

    std::vector<zfxvariant> GeometryObject::getPointAttr(const std::string& attr_name) const {
        std::vector<zfxvariant> res;
        res.resize(m_points.size());
        for (int i = 0; i < m_points.size(); i++)
        {
            assert(m_points[i] && m_points[i]->attr.find(attr_name) != m_points[i]->attr.end());
            res[i] = m_points[i]->attr[attr_name];
        }
        return res;
    }

    bool GeometryObject::deletePointAttr(const std::string& attr_name) {
        for (int i = 0; i < m_points.size(); i++)
        {
            assert(m_points[i] && m_points[i]->attr.find(attr_name) != m_points[i]->attr.end());
            m_points[i]->attr.erase(attr_name);
        }
        return false;
    }

    int GeometryObject::addpoint(zfxvariant var) {
        vec3f pos = std::visit([](auto&& val)->vec3f {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, glm::vec3>) {
                return vec3f({ val[0],val[1], val[2] });
            }
            else if constexpr (std::is_same_v<T, zfxfloatarr>) {
                return vec3f({ val[0],val[1], val[2] });
            }
            else {
                throw makeError<UnimplError>("type dismatch");
            }
        }, var);

        auto spPoint = std::make_shared<Point>();
        spPoint->pos = pos;
        m_points.emplace_back(spPoint);
        return m_points.size() - 1;
    }

    int GeometryObject::addvertex(int face_id, int point_id) {
        if (face_id < 0 || face_id >= m_faces.size() ||
            point_id < 0 || point_id >= m_points.size()) {
            return -1;
        }
        auto& spPoint = m_points[point_id];
        if (spPoint->edges.empty()) {
            auto pFace = m_faces[face_id];
            assert(pFace);
            auto firstH = pFace->h;
            auto& [prevPoint, prevEdge, pprevEdge] = getPrev(firstH);
            auto newedge = std::make_shared<HEdge>();
            newedge->point = prevEdge->point;
            newedge->next = firstH;
            newedge->face = face_id;
            newedge->id = generateUUID();
            prevEdge->next = newedge.get();
            prevEdge->point = point_id;
            m_hEdges.insert(std::make_pair(newedge->id, newedge));
            m_bTriangle = false;    //��������ζ�Ų��ܵ��������δ�����
            return nvertices(face_id);
        }
        else {
            return -1;
        }
    }

    int GeometryObject::nvertices(int face_id) const {
        if (face_id < 0 || face_id >= m_faces.size()) {
            return 0;
        }
        auto firstedge = m_faces[face_id]->h;
        auto h = firstedge;
        int ncount = 0;
        do {
            ncount++;
            h = h->next;
        } while (firstedge != h);
        return ncount;
    }

    void GeometryObject::addprim() {

    }

    bool GeometryObject::remove_faces(const std::set<int>& faces, bool includePoints) {
        if (faces.empty()) {
            return false;
        }

        std::set<int> remPoints;
        std::set<Point*> remPtrPoints;
        for (auto face_id : faces) {
            auto pFace = m_faces[face_id];
            if (includePoints) {
                //����face�����еĵ㣬�Ƿ��е���Ҫ��ɾ��
                auto firsth = pFace->h;
                auto h = firsth;
                do 
                {
                    auto& spPoint = m_points[h->point];
                    spPoint->edges.erase(h);
                    if (spPoint->edges.empty()) {
                        remPoints.insert(h->point);
                        remPtrPoints.insert(spPoint.get());
                    }
                    if (h->pair) {
                        //�����ÿ��Լ�
                        h->pair->pair = nullptr;
                    }
                    h->pair = nullptr;
                    auto nexth = h->next;
                    //h������������ˣ����ھͿ���ɾ���ˡ�
                    m_hEdges.erase(h->id);
                    h = nexth;
                } while (firsth != h);
            }
        }

        //�߶��Ѿ�ɾ���ˣ�����ֻҪɾ������漴�ɡ�
        removeElements(m_points, remPoints);
        removeElements(m_faces, faces);


        //��Ϊ�㱻ɾ���ˣ�����Ҫ��������δ��ɾ���ıߣ�ֻҪ����point��face��������������face����Ҫ��������Ϊɾ���ľ���face��ֱ��ɾ������
        for (auto& [_, hedge] : m_hEdges) {
            int nStep = 0;
            for (auto remPointId : remPoints) {
                if (hedge->point >= remPointId)
                    nStep++;
                else
                    break;
            }
            hedge->point -= nStep;

            nStep = 0;
            for (auto remFaceId : faces) {
                if (hedge->face >= remFaceId)
                    nStep++;
                else
                    break;
            }
            hedge->face -= nStep;

            auto ph = hedge.get();
            auto fh = m_faces[hedge->face]->h;
            if (m_bTriangle) {
                assert(fh == ph || fh == ph->next || fh == ph->next->next);
            }
        }
        return true;
    }

    int GeometryObject::npoints() const {
        return m_points.size();
    }

    int GeometryObject::nfaces() const {
        return m_faces.size();
    }

    int GeometryObject::nvertices() const {
        //TODO:
        return -1;
    }

}