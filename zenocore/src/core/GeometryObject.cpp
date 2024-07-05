#include <zeno/types/GeometryObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <assert.h>


namespace zeno
{
    ZENO_API GeometryObject::GeometryObject() {

    }

    ZENO_API GeometryObject::GeometryObject(const GeometryObject& geo) {

    }

    ZENO_API GeometryObject::GeometryObject(PrimitiveObject* prim) {
        initFromPrim(prim);
    }

    ZENO_API std::shared_ptr<PrimitiveObject> GeometryObject::toPrimitive() const {
        std::shared_ptr<PrimitiveObject> spPrim = std::make_shared<PrimitiveObject>();
        spPrim->verts->resize(m_points.size());
        for (int i = 0; i < m_points.size(); i++) {
            spPrim->verts[i] = m_points[i].pos;
            //TODO: attr
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
            //TODO: 两种大小都存在的情况
            if (points.size() == 4) {

            }
            else if (points.size() == 3) {
                if (spPrim->tris.size() == 0) {
                    spPrim->tris->resize(m_faces.size());
                }
                vec3i tri = { points[0],points[1], points[2] };
                spPrim->tris[i] = std::move(tri);
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

    void GeometryObject::initFromPrim(PrimitiveObject* prim) {

        m_points.resize(prim->verts->size());
        for (int i = 0; i < m_points.size(); i++) {
            m_points[i].pos = prim->verts[i];
        }

        int nFace = -1;
        bool bTriangle = prim->loops->empty() && !prim->tris->empty();
        if (bTriangle) {
            nFace = prim->tris->size();
            m_hEdges.reserve(nFace * 3);
        }
        else {
            assert(!prim->loops->empty() && !prim->polys->empty());
            nFace = prim->polys->size();
            //一般是四边形
            m_hEdges.reserve(nFace * 4);
        }

        for (int face = 0; face < nFace; face++) {
            std::vector<int> points;
            if (bTriangle) {
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
                }

                m_hEdges.emplace_back(hedge);
                m_points[vp].edges.insert(hpq);

                lastHedge = hpq;
            }
        }
    }

}