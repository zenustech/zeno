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

    int GeometryObject::visit_allHalfEdge_from(int fromPoint, std::function<bool(int)> f) {
        assert(fromPoint < m_points.size());
        int firstH = m_points[fromPoint].hEdge, h = firstH;
        do {
            const auto& hedge = m_hEdges[h];
            h = m_hEdges[hedge.pair].next; //h.pair.next
            int ret = -1;
            if (f(h)) {
                return ret;
            }
        } while (h != firstH);
    }

    int GeometryObject::checkHEdge(int fromPoint, int toPoint) {
        assert(fromPoint < m_points.size());
        int firstH = m_points[fromPoint].hEdge, h = firstH;
        do {
            if (h == -1)
                return -1;
            const auto& hedge = m_hEdges[h];
            h = m_hEdges[hedge.pair].next; //h.pair.next
            if (m_hEdges[h].point == toPoint) {
                return h;
            }
        } while (h != firstH);
    }

    int GeometryObject::getNextOutEdge(int fromPoint, int currentOutEdge) {
        assert(fromPoint < m_points.size());
        int firstH = m_points[fromPoint].hEdge, h = firstH;
        do {
            const auto& hedge = m_hEdges[h];
            int hNext = m_hEdges[hedge.pair].next; //h.pair.next
            int ret = -1;
            if (h == currentOutEdge)
                return hNext;
            h = hNext;
        } while (h != firstH);
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

            int lastHedge = -1;
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
                int hpq = checkHEdge(vp, vq);
                if (hpq == -1) {
                    HEdge hedge;
                    hedge.face = face;
                    hedge.point = vq;
                    hpq = m_hEdges.size();
                    if (lastHedge != -1) {
                        m_hEdges[lastHedge].next = hpq;
                    }
                    if (i == points.size() - 1) {
                        hedge.next = m_points[points[0]].hEdge;
                    }
                    m_hEdges.emplace_back(hedge);
                }
                m_points[vp].hEdge = hpq;

                lastHedge = hpq;

                // vq->vp
                int hqp = checkHEdge(vq, vp);
                if (hqp == -1) {
                    hqp = m_hEdges.size();
                    m_hEdges.emplace_back(HEdge());

                    HEdge& hedge = m_hEdges[hqp];
                    hedge.face = -1;    //unknown face, may be a hole.
                    hedge.point = vp;
                    hedge.pair = hpq;
                    hedge.next = hpq;

                    m_hEdges[hpq].pair = hqp;
                    hedge.next = getNextOutEdge(vp, hpq);
                }
            }
        }
    }
}