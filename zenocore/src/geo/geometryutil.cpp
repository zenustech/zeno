#include <zeno/geo/geometryutil.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/helper.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/para/parallel_reduce.h>
#include <zeno/geo/kdsearch.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <random>
#include <zeno/utils/wangsrng.h>
#include <unordered_set>
#include <deque>
#include <zeno/utils/interfaceutil.h>
#include "nanoflann.hpp"
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno
{
    // KD-Tree 适配器
    struct _PointCloud {
        std::vector<vec3f> pts;

        inline size_t kdtree_get_point_count() const { return pts.size(); }
        inline double kdtree_get_pt(const size_t idx, int dim) const { return pts[idx][dim]; }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    };

    // 并查集实现
    class UnionFind {
    private:
        std::vector<int> parent, rank;
    public:
        explicit UnionFind(int n) : parent(n), rank(n, 0) {
            for (int i = 0; i < n; ++i) parent[i] = i;
        }
        int find(int x) {
            if (parent[x] == x) return x;
            return parent[x] = find(parent[x]);
        }
        void unite(int x, int y) {
            int rootX = find(x), rootY = find(y);
            if (rootX == rootY) return;
            if (rank[rootX] < rank[rootY]) std::swap(rootX, rootY);
            parent[rootY] = rootX;
            if (rank[rootX] == rank[rootY]) rank[rootX]++;
        }
    };

    bool prim_remove_point(GeometryObject* prim, int ptnum)
    {
        return false;
    }

    ZENO_API std::vector<vec3f> calc_point_normals(GeometryObject* geo, const std::vector<vec3i>& tris, float flip)
    {
        size_t n = geo->npoints();
        std::vector<vec3f> normals(n);

        std::vector<vec3f> pos(geo->points_pos());

        for (size_t i = 0; i < tris.size(); i++) {
            auto ind = tris[i];
            size_t i1 = ind[0], i2 = ind[1], i3 = ind[2];
            vec3f pos1(pos[i1]);
            vec3f pos2(pos[i2]);
            vec3f pos3(pos[i3]);
            auto n = cross(pos2 - pos1, pos3 - pos1);
            normals[i1] += n;
            normals[i2] += n;
            normals[i3] += n;
        }
        for (size_t i = 0; i < normals.size(); i++) {
            normals[i] = flip * normalizeSafe(normals[i]);
        }

        return normals;
    }

    ZENO_API std::vector<vec3f> calc_triangles_tangent(
        GeometryObject* geo,
        bool has_uv,
        const std::vector<vec3i>& tris,
        const std::vector<vec3f>& pos,
        const std::vector<vec3f>& uvarray,
        const std::vector<vec3f>& uv0data,
        const std::vector<vec3f>& uv1data,
        const std::vector<vec3f>& uv2data
    ) {
        //有可能tang已经计算出来的
        std::vector<vec3f> tang(tris.size());
        if (has_uv) {
#pragma omp parallel for
            for (auto i = 0; i < tris.size(); ++i) {
                const auto& pos0 = pos[tris[i][0]];
                const auto& pos1 = pos[tris[i][1]];
                const auto& pos2 = pos[tris[i][2]];
                zeno::vec3f uv0;
                zeno::vec3f uv1;
                zeno::vec3f uv2;

                uv0 = uv0data[i];
                uv1 = uv1data[i];
                uv2 = uv2data[i];

                auto edge0 = pos1 - pos0;
                auto edge1 = pos2 - pos0;
                auto deltaUV0 = uv1 - uv0;
                auto deltaUV1 = uv2 - uv0;

                auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

                zeno::vec3f tangent;
                tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
                tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
                tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
                //printf("tangent:%f %f %f\n", tangent[0], tangent[1], tangent[2]);
                //zeno::log_info("tangent {} {} {}",tangent[0], tangent[1], tangent[2]);
                auto tanlen = zeno::length(tangent);
                tangent* (1.f / (tanlen + 1e-8));
                /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
                    zeno::vec3f n = nrm[tris[i][0]], unused;
                    zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
                } else {
                    tang[i] = tangent * (1.f / tanlen);
                }*/
                tang[i] = tangent;
            }
        }
        else {
#pragma omp parallel for
            for (auto i = 0; i < tris.size(); ++i) {
                const auto& pos0 = pos[tris[i][0]];
                const auto& pos1 = pos[tris[i][1]];
                const auto& pos2 = pos[tris[i][2]];
                zeno::vec3f uv0;
                zeno::vec3f uv1;
                zeno::vec3f uv2;

                uv0 = uvarray[tris[i][0]];
                uv1 = uvarray[tris[i][1]];
                uv2 = uvarray[tris[i][2]];

                auto edge0 = pos1 - pos0;
                auto edge1 = pos2 - pos0;
                auto deltaUV0 = uv1 - uv0;
                auto deltaUV1 = uv2 - uv0;

                auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

                zeno::vec3f tangent;
                tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
                tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
                tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
                //printf("tangent:%f %f %f\n", tangent[0], tangent[1], tangent[2]);
                //zeno::log_info("tangent {} {} {}",tangent[0], tangent[1], tangent[2]);
                auto tanlen = zeno::length(tangent);
                tangent* (1.f / (tanlen + 1e-8));
                /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
                    zeno::vec3f n = nrm[tris[i][0]], unused;
                    zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
                    } else {
                    tang[i] = tangent * (1.f / tanlen);
                    }*/
                tang[i] = tangent;
            }
        }
        return tang;
    }

    ZENO_API std::vector<vec3f> compute_vertex_tangent(
        const std::vector<vec3i>& tris,
        const std::vector<vec3f>& tang,
        const std::vector<vec3f>& pos
    ) {
        std::vector<vec3f> atang;
        atang.assign(atang.size(), zeno::vec3f(0));
        for (size_t i = 0; i < tris.size(); ++i)
        {

            auto vidx = tris[i];
            zeno::vec3f v0 = pos[vidx[0]];
            zeno::vec3f v1 = pos[vidx[1]];
            zeno::vec3f v2 = pos[vidx[2]];
            auto e1 = v1 - v0, e2 = v2 - v0;
            float area = zeno::length(zeno::cross(e1, e2)) * 0.5;
            atang[vidx[0]] += area * tang[i];
            atang[vidx[1]] += area * tang[i];
            atang[vidx[2]] += area * tang[i];
        }
#pragma omp parallel for
        for (auto i = 0; i < atang.size(); i++)
        {
            atang[i] = atang[i] / (length(atang[i]) + 1e-6);
        }
        return atang;
    }

    ZENO_API std::pair<vec3f, vec3f> geomBoundingBox(GeometryObject* geo) {
        const std::vector<vec3f>& verts = geo->points_pos();
        if (!verts.size())
            return { {0, 0, 0}, {0, 0, 0} };
        return parallel_reduce_minmax(verts.begin(), verts.end());
    }

    std::optional<std::pair<vec3f, vec3f>> geomBoundingBox2(GeometryObject* geo) {
        const std::vector<vec3f>& verts = geo->points_pos();
        if (!verts.size())
            return std::nullopt;
        return parallel_reduce_minmax(verts.begin(), verts.end());
    }

    void geom_set_abcpath(GeometryObject_Adapter* geom, zeno::String path_name) {
        auto pUserData = geom->userData();
        int faceset_count = pUserData->get_int("abcpath_count", 0);
        for (auto j = 0; j < faceset_count; j++) {
            pUserData->del(stdString2zs(zeno::format("abcpath_{}", j)));
        }
        pUserData->set_int("abcpath_count", 1);
        pUserData->set_string("abcpath_0", path_name);
        geom->create_face_attr("abcpath", (int)0);
    }

    void geom_set_faceset(GeometryObject_Adapter* geom, zeno::String faceset_name) {
        IUserData* pUserData = geom->userData();
        int faceset_count = pUserData->get_int("faceset_count", 0);
        for (auto j = 0; j < faceset_count; j++) {
            pUserData->del(stdString2zs(zeno::format("faceset_{}", j)));
        }
        pUserData->set_int("faceset_count", 1);
        pUserData->set_string("faceset_0", faceset_name);
        geom->create_face_attr("faceset", (int)0);
    }

    static void set_special_attr_remap(GeometryObject_Adapter* p, std::string attr_name, std::unordered_map<std::string, int>& facesetNameMap) {
        UserData* pUserData = dynamic_cast<UserData*>(p->userData());
        int faceset_count = pUserData->get_int(stdString2zs(attr_name + "_count"), 0);
        {
            std::unordered_map<int, int> cur_faceset_index_map;
            for (auto i = 0; i < faceset_count; i++) {
                auto path = pUserData->get2<std::string>(format("{}_{}", attr_name, i));
                if (facesetNameMap.count(path) == 0) {
                    int new_index = facesetNameMap.size();
                    facesetNameMap[path] = new_index;
                }
                cur_faceset_index_map[i] = facesetNameMap[path];
            }

            for (int i = 0; i < p->nfaces(); i++) {
                int val = p->m_impl->get_elem<int>(ATTR_FACE, attr_name, 0, i);
                int newval = cur_faceset_index_map[val];
                p->m_impl->set_elem(ATTR_FACE, attr_name, i, newval);
            }
        }
    }


    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> mergeObjects(
        zeno::ListObject* spList,
        std::string const& tagAttr,
        bool tag_on_vert,
        bool tag_on_face)
    {
        size_t nTotalPts = 0, nTotalFaces = 0, nVertices = 0;
        const std::vector<zeno::GeometryObject_Adapter*>& geoobjs = spList->m_impl->get<zeno::GeometryObject_Adapter>();
        for (auto spObject : geoobjs) {
            nTotalPts += spObject->npoints();
            nTotalFaces += spObject->nfaces();
        }

        std::vector<zeno::vec3f> newObjPos(nTotalPts);
        std::vector<std::vector<int>> newFaces(nTotalFaces);

        size_t pt_offset = 0, face_offset = 0;
        bool bTriangle = true;
        for (int iGeom = 0; iGeom < geoobjs.size(); iGeom++)
        {
            auto elemObj = geoobjs[iGeom];
            std::vector<vec3f> obj_pos = elemObj->m_impl->points_pos();
            int nPts = elemObj->npoints();
            int nFaces = elemObj->nfaces();
            if (!elemObj->is_base_triangle()) {
                bTriangle = false;
            }

            std::copy(obj_pos.begin(), obj_pos.end(), newObjPos.begin() + pt_offset);
            for (int iFace = 0; iFace < nFaces; iFace++) {
                std::vector<int> facePoints = elemObj->m_impl->face_points(iFace);
                for (int k = 0; k < facePoints.size(); ++k) {
                    facePoints[k] += pt_offset;
                }
                newFaces[iFace + face_offset] = std::move(facePoints);
            }
            pt_offset += nPts;
            face_offset += nFaces;
        }

        auto mergedObj = create_GeometryObject(zeno::Topo_IndiceMesh, bTriangle, newObjPos, newFaces);
        //copy attribute from each obj
        pt_offset = 0;
        face_offset = 0;
        for (int iGeom = 0; iGeom < geoobjs.size(); iGeom++) {
            auto elemObj = geoobjs[iGeom];
            int nPts = elemObj->npoints();
            int nFaces = elemObj->nfaces();
            mergedObj->inheritAttributes(elemObj, -1, pt_offset, { "pos" }, face_offset, {});
            pt_offset += nPts;
            face_offset += nFaces;
        }

        mergedObj->create_attr(ATTR_POINT, "pos", newObjPos);

        std::vector<std::string> matNameList(0);
        std::unordered_map<std::string, int> facesetNameMap;
        std::unordered_map<std::string, int> abcpathNameMap;
        for (auto p : geoobjs) {
            //if p has material
            int matNum = p->userData()->get_int("matNum", 0);
            if (matNum > 0) {
                for (int i = 0; i < p->nfaces(); i++) {
                     int matid = p->m_impl->get_elem<int>(ATTR_FACE, "matid", 0, i);
                     if (matid != -1) {
                         p->m_impl->set_elem<int>(ATTR_FACE, "matid", i, matid + 1);
                     }
                }
                for (int i = 0; i < matNum; i++) {
                    auto matIdx = "Material_" + to_string(i);
                    auto matName = p->userData()->get_string(stdString2zs(matIdx), "Default");
                    matNameList.emplace_back(zsString2Std(matName));
                }
            }
            else {
                if (p->nfaces() > 0) {
                    p->set_face_attr("matid", (int)-1);
                }
            }

            int faceset_count = p->userData()->get_int("faceset_count", 0);
            if (faceset_count == 0) {
                geom_set_faceset(p, "defFS");
                faceset_count = 1;
            }
            set_special_attr_remap(p, "faceset", facesetNameMap);

            int path_count = p->userData()->get_int("abcpath_count", 0);
            if (path_count == 0) {
                geom_set_abcpath(p, "/ABC/unassigned");
                path_count = 1;
            }
            set_special_attr_remap(p, "abcpath", abcpathNameMap);
        }

        //合并UserData
        auto pUserData = dynamic_cast<UserData*>(mergedObj->userData());
        for (auto& p : geoobjs) {
            pUserData->merge(*dynamic_cast<UserData*>(p->userData()));
        }

        if (matNameList.size() > 0) {
            //add matNames to userData
            int i = 0;
            for (auto name : matNameList) {
                auto matIdx = "Material_" + to_string(i);
                pUserData->setLiterial(matIdx, name);
                i++;
            }
        }
        int oMatNum = matNameList.size();
        pUserData->set2("matNum", oMatNum);
        {
            for (auto const& [k, v] : facesetNameMap) {
                pUserData->set2(format("faceset_{}", v), k);
            }
            int faceset_count = facesetNameMap.size();
            pUserData->set2("faceset_count", faceset_count);
        }
        {
            for (auto const& [k, v] : abcpathNameMap) {
                pUserData->set2(format("abcpath_{}", v), k);
            }
            int abcpath_count = abcpathNameMap.size();
            pUserData->set2("abcpath_count", abcpath_count);
        }

        return mergedObj;
    }

    ZENO_API bool dividePlane(
        const std::vector<vec3f>& face_pts, /*容器的顺序就是面点的逆序排序*/
        float A, float B, float C, float D,
        /*out*/std::vector<DivideFace>& split_faces,
        /*out*/std::map<int, DividePoint>& split_infos,
        /*out*/PointSide& which_side_if_failed
    ) {
        //TODO: 先用bbox过滤
        {
            bool hasbelow = false, hasabove = false;
            for (auto pos : face_pts) {
                float metric = A * pos[0] + B * pos[1] + C * pos[2] + D;
                if (metric < 0) {
                    hasbelow = true;
                }
                else if (metric > 0) {
                    hasabove = true;
                }
            }
            if (!(hasbelow && hasabove)) {
                which_side_if_failed = Both;
                if (hasbelow) which_side_if_failed = Below;
                if (hasabove) which_side_if_failed = Above;
                return false;
            }
        }

        int nCount = face_pts.size();
        std::vector<int> splitters;   //储存分割点的序号，注意，分割点也可以是顶点的序号
        std::vector<int> newface_pt_seq;  //原有的点加上分割点形成的点序号序列，后续要遍历这个序列得到分割面
        std::map<int, vec3f> splitter_pos;
        for (int i = 1; i <= face_pts.size(); i++) {
            //观察Pi-1和Pi之间是否有分割
            int from = -1, to = -1;
            if (i == face_pts.size()) {
                //TODO: 如果是线段，就不考虑最后的闭合线
                from = face_pts.size() - 1;
                to = 0;
            }
            else {
                from = i - 1;
                to = i;
            }
            vec3f p1 = face_pts[from], p2 = face_pts[to];
            float d1 = A * p1[0] + B * p1[1] + C * p1[2] + D;
            float d2 = A * p2[0] + B * p2[1] + C * p2[2] + D;
            newface_pt_seq.push_back(from);
            if (d1 * d2 < 0) {
                //新增一个分割点
                int new_splitter = nCount++;
                splitters.push_back(new_splitter);
                newface_pt_seq.push_back(new_splitter);
                float t = -(A*p1[0] + B*p1[1] + C*p1[2] + D) / (A*(p2[0]-p1[0]) + B*(p2[1]-p1[1]) + C * (p2[2]-p1[2]));
                float newx = p1[0] + t * (p2[0] - p1[0]);
                float newy = p1[1] + t * (p2[1] - p1[1]);
                float newz = p1[2] + t * (p2[2] - p1[2]);
                vec3f spliter_pos(newx, newy, newz);
                splitter_pos.insert(std::make_pair(new_splitter, spliter_pos));
                split_infos.insert(std::make_pair(new_splitter, DividePoint{ spliter_pos, from, to}));
            }
            else if (d1 * d2 == 0) {
                int new_splitter;
                if (d1 == 0) {
                    new_splitter = from;
                    if (split_infos.find(new_splitter) == split_infos.end())
                    {
                        //顶点也是分割点的情况，会重复储存（因为当前边和下一条边都会访问到），所以要判断是否已经存了
                        splitters.push_back(new_splitter);
                        splitter_pos.insert(std::make_pair(new_splitter, p1));
                        split_infos.insert(std::make_pair(new_splitter, DividePoint{ p1, from, from }));
                    }
                }
                if (d2 == 0) {
                    new_splitter = to;
                    if (split_infos.find(new_splitter) == split_infos.end())
                    {
                        splitters.push_back(new_splitter);
                        splitter_pos.insert(std::make_pair(new_splitter, p2));
                        split_infos.insert(std::make_pair(new_splitter, DividePoint{ p2, to, to }));
                    }
                }
            }
        }

        std::vector<int> next_pts(nCount);
        for (int i = 1; i < nCount; i++) {
            if (i == 0) {
                next_pts[newface_pt_seq[nCount - 1]] = newface_pt_seq[0];
            }
            next_pts[newface_pt_seq[i - 1]] = newface_pt_seq[i];
        }

        assert(splitters.size() > 1 && splitter_pos.size() > 1);

        //要对分割点进行排序，以便于在分割点之间“跳跃”
        {
            //先确定方向，
            vec3f sp1 = splitter_pos.begin()->second;
            vec3f sp2 = splitter_pos.rbegin()->second;
            vec3f d = sp2 - sp1;
            float _x = std::abs(d[0]), _y = std::abs(d[1]), _z = std::abs(d[2]);
            int proj = 0;
            if (_x < _y) {
                if (_y < _z) {
                    proj = 2;
                }
                else {
                    proj = 1;
                }
            }
            else {
                if (_x < _z) {
                    proj = 2;
                }
                else {
                    proj = 0;
                }
            }
            std::sort(splitters.begin(), splitters.end(), [&](int splitter1, int splitter2) {
                return splitter_pos[splitter1][proj] < splitter_pos[splitter2][proj];
            });
        }

        //找到要遍历的起点（这些起点不能是分割点，如果分割点和顶点重合，也不能作为起点）
        std::deque<int> start_pts;
        for (int i = 0; i < face_pts.size(); i++) {
            if (std::find(splitters.begin(), splitters.end(), i) == splitters.end()) {
                start_pts.push_back(i);
            }
        }

        std::function<bool(int, PointSide, std::vector<int>& )> search_ring = [&](
                int currpt, 
                PointSide side,
                std::vector<int>& exist_pts)->bool {
            //当前处在currpt点，已经收集到的点序（按逆时针排序）为exist_pts(不包含currpt)

            //首先判断是否闭环
            if (!exist_pts.empty() && next_pts[currpt] == exist_pts[0]) {
                exist_pts.push_back(currpt);
                return true;
            }

            //其次判断是否有重复路径
            if (std::find(exist_pts.begin(), exist_pts.end(), currpt) != exist_pts.end())
                return false;

            std::vector<int> temp_pts = exist_pts;
            temp_pts.push_back(currpt);
            int pt = next_pts[currpt];
            
            while (true) {
                //首先判断pt是否分割点
                vec3f ptpos;
                auto iterSpliter = splitter_pos.find(pt);
                if (iterSpliter != splitter_pos.end()) {
                    temp_pts.push_back(pt);
                    break;
                }
                else {
                    ptpos = face_pts[pt];
                }
                float metric = A * ptpos[0] + B * ptpos[1] + C * ptpos[2];
                if (metric < 0 && side == Above ||
                    metric > 0 && side == Below) {
                    //应该是先遇到分割点，而不是跳到另一侧
                    assert(false);
                    break;
                }
                //从currpt到pt-1，都是同一侧的点，可以加入
                temp_pts.push_back(pt);
                pt = next_pts[pt];      //取下一个点
            }

            //pt是一个分割点
            int split_pt = pt;
            //找pt的前一个分割点和后一个分割点，然后各自搜索下去，看能否形成闭环
            int split_idx = std::find(splitters.begin(), splitters.end(), split_pt) - splitters.begin();
            bool ret = false;
            if (split_idx > 0) {
                int prev_split_pt = splitters[split_idx - 1];
                ret = search_ring(prev_split_pt, side, temp_pts);
                if (ret) {
                    exist_pts = temp_pts;
                    return ret;
                }
            }
            if (split_idx < splitters.size() - 1) {
                int next_split_pt = splitters[split_idx + 1];
                ret = search_ring(next_split_pt, side, temp_pts);
                if (ret) {
                    exist_pts = temp_pts;
                    return true;
                }
            }
            return ret;
        };

        //遍历所有非分割点
        std::set<int> visited;
        assert(!start_pts.empty());
        while (!start_pts.empty())
        {
            int start_pt = start_pts.front();
            start_pts.pop_front();
            if (visited.find(start_pt) != visited.end()) {
                //这个非分割点已经被加到某一个面里了，不可能属于另一个面
                continue;
            }

            vec3f start_pos = face_pts[start_pt];

            float metric = A * start_pos[0] + B * start_pos[1] + C * start_pos[2] + D;
            assert(metric != 0);

            PointSide side = UnDecided;
            if (metric < 0) side = Below;
            else side = Above;

            std::vector<int> exist_face;
            bool ret = search_ring(start_pt, side, exist_face);
            if (ret) {
                DivideFace divideface;
                divideface.face_indice = exist_face;
                divideface.side = side;
                split_faces.push_back(divideface);
            }

            for (auto pid : exist_face) {
                visited.insert(pid);
            }
        }

#if 0
        int nPoints = face_pts.size();
        std::unordered_set<int> added_points;
        //开始循环遍历newface_pt_seq，不断收集点以形成新面
        while (added_points.size() < face_pts.size()) {
            PointSide side = UnDecided;
            std::vector<int> new_face;
            for (int i = 0; ; i++) {
                int curr_pt = newface_pt_seq[i];
                if (!new_face.empty() && curr_pt == new_face[0]) {
                    break;
                }
                vec3f curr_pos = face_pts[curr_pt];
                if (added_points.find(curr_pt) != added_points.end()) {
                    continue;
                }
                float metric = A * curr_pos[0] + B * curr_pos[1] + C * curr_pos[2] + D;
                if (side == UnDecided) {
                    if (metric < 0) {
                        side = Below;
                    }
                    else if (metric > 0) {
                        side = Above;
                    }
                    else {
                        //这里的顶点和分割点重合，仍未决定哪一边，等下一圈再跑一边
                        break;
                    }
                }
                if (metric < 0 && side == Below) {
                    new_face.push_back(curr_pt);
                    added_points.insert(curr_pt);
                }
                else if (metric > 0 && side == Above) {
                    new_face.push_back(curr_pt);
                    added_points.insert(curr_pt);
                }
                else if (metric == 0) {
                    //此时是分割点
                    new_face.push_back(curr_pt);
                }
            }
            split_faces.push_back(new_face);
        }
#endif
        return true;
    }

    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> divideObject(
        zeno::GeometryObject_Adapter* input_object,
        DivideKeep keep,
        zeno::vec3f center_pos,
        zeno::vec3f direction
    )
    {
        const auto& pos = input_object->m_impl->points_pos();
        const int nface = input_object->m_impl->nfaces();
        std::vector<std::vector<int>> newFaces;
        std::vector<vec3f> new_pos = pos;

        float A = direction[0], B = direction[1], C = direction[2],
            D = -(A * center_pos[0] + B * center_pos[1] + C * center_pos[2]);

        int nPoints = pos.size();
        std::map<std::string, int> split_cache;
        std::set<int> rem_points;

        for (int iFace = 0; iFace < nface; iFace++) {
            std::vector<int> face_indice = input_object->m_impl->face_points(iFace);
            std::vector<vec3f> face_pos(face_indice.size());
            for (int i = 0; i < face_indice.size(); i++)
            {
                face_pos[i] = pos[face_indice[i]];
            }
            std::vector<DivideFace> split_faces;
            std::map<int, DividePoint> split_infos;     //key: 相对索引，这个索引值是“内部的”
            PointSide whichside_iffailed = UnDecided;
            bool bSuccess = dividePlane(face_pos, A, B, C, D, split_faces, split_infos, whichside_iffailed);
            if (!bSuccess) {
                //不需要分割
                if (keep == Keep_Both) {
                    newFaces.push_back(face_indice);
                }
                else if (keep == Keep_Below) {
                    if (whichside_iffailed == Below) {
                        newFaces.push_back(face_indice);
                    }
                    else {
                        for (auto pt : face_indice)
                            rem_points.insert(pt);
                    }
                }
                else if (keep == Keep_Above) {
                    if (whichside_iffailed == Above) {
                        newFaces.push_back(face_indice);
                    }
                    else {
                        for (int pt : face_indice)
                            rem_points.insert(pt);
                    }
                }
                continue;
            }

            for (const DivideFace& divideFace : split_faces)
            {
                //new_face_indice是“相对索引”，现在要转为基于全局points的索引。
                std::vector<int> new_face_abs_indice;
                for (int relidx : divideFace.face_indice) {
                    int final_idx = -1;
                    if (relidx < face_indice.size()) {
                        final_idx = face_indice[relidx];
                    }
                    else {
                        //新增的分割点，需要查表看是不是缓存了
                        DividePoint& split_info = split_infos[relidx];
                        int rel_p1 = split_info.from, rel_p2 = split_info.to;
                        int abs_p1 = face_indice[rel_p1], abs_p2 = face_indice[rel_p2];
                        std::string key = zeno::format("{}->{}", std::min(abs_p1, abs_p2), std::max(abs_p1, abs_p2));
                        auto iter = split_cache.find(key);
                        if (iter != split_cache.end()) {
                            final_idx = iter->second;
                        }
                        else {
                            final_idx = new_pos.size();
                            new_pos.push_back(split_info.pos);
                            split_cache.insert(std::make_pair(key, final_idx));
                        }
                    }
                    new_face_abs_indice.push_back(final_idx);
                }

                //这里要看保留哪部分
                if (keep == Keep_Both) {
                    newFaces.emplace_back(std::move(new_face_abs_indice));
                }
                else if (keep == Keep_Below) {
                    if (divideFace.side == Below) {
                        newFaces.emplace_back(std::move(new_face_abs_indice));
                    }
                    else {
                        for (int relidx : divideFace.face_indice) {
                            if (split_infos.find(relidx) == split_infos.end()) {
                                int rempt = face_indice[relidx];
                                rem_points.insert(rempt);
                            }
                        }
                    }
                }
                else if (keep == Keep_Above) {
                    if (divideFace.side == Above) {
                        newFaces.emplace_back(std::move(new_face_abs_indice));
                    }
                    else {
                        for (int relidx : divideFace.face_indice) {
                            if (split_infos.find(relidx) == split_infos.end()) {
                                int rempt = face_indice[relidx];
                                rem_points.insert(rempt);
                            }
                        }
                    }
                }
            }
        }

        //todo: 考虑线
        auto spOutput = create_GeometryObject(zeno::Topo_IndiceMesh, false, new_pos, newFaces);

        //去除被排除的部分
        for (auto iter = rem_points.rbegin(); iter != rem_points.rend(); iter++) {
            spOutput->m_impl->remove_point(*iter);
        }
        return spOutput;
    }

    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> scatter(
        zeno::GeometryObject_Adapter* input,
        const std::string& sampleRegion,
        const int nPointCount,
        int seed)
    {
        auto spOutput = create_GeometryObject(zeno::Topo_IndiceMesh, input->is_base_triangle(), nPointCount, 0);
        if (seed == -1) seed = std::random_device{}();

        const int nFaces = input->m_impl->nfaces();
        const std::vector<vec3f>& pos = input->m_impl->points_pos();
        std::vector<float> cdf(nFaces);
        std::vector<vec3f> newPts(nPointCount);

        if (sampleRegion == "Volumn") {
            const auto& bbox = geomBoundingBox(input->m_impl.get());
            //目前只考虑通过bbox直接采样，不考虑复杂的空间
            parallel_for((size_t)0, (size_t)nPointCount, [&](size_t i) {
                wangsrng rng(i);
                float rx = rng.next_float();
                float ry = rng.next_float();
                float rz = rng.next_float();
                float xp = bbox.first[0] + (bbox.second[0] - bbox.first[0]) * rx;
                float yp = bbox.first[1] + (bbox.second[1] - bbox.first[1]) * ry;
                float zp = bbox.first[2] + (bbox.second[2] - bbox.first[2]) * rz;
                //TODO: 如果面对复杂的非矩形几何体，可以通过判断点是否在几何体内部。
                newPts[i] = vec3f(xp, yp, zp);
            });
            spOutput->create_point_attr("pos", newPts);
            return spOutput;
        }

        std::vector<std::vector<int>> faces = input->m_impl->face_indice();
        parallel_inclusive_scan_sum(faces.begin(), faces.end(), cdf.begin(), [&](std::vector<int> const& ind) {
            auto i1 = ind[0];
            auto i2 = ind[1];
            auto i3 = ind[2];
            auto i4 = ind[3];
            auto area = length(cross(pos[i3] - pos[i2], pos[i2] - pos[i1]));
            return area;
        });

        auto inv_total = 1 / cdf.back();
        parallel_for((size_t)0, cdf.size(), [&](size_t i) {
            cdf[i] *= inv_total;
        });

        if (false/*line*/) {
            //todo
        }
        else {
            parallel_for((size_t)0, (size_t)nPointCount, [&](size_t i) {
                wangsrng rng(seed, i);
                auto val = rng.next_float();
                auto it = std::lower_bound(cdf.begin(), cdf.end(), val);
                size_t index = it - cdf.begin();
                assert(index < nFaces);
                auto const& pointsIndice = faces[index];
                auto r1 = std::sqrt(rng.next_float());
                auto r2 = rng.next_float();
                if (pointsIndice.size() == 4) {
                    //目前只考虑直角四边形或平行四边形
                    vec3f v01 = pos[pointsIndice[1]] - pos[pointsIndice[0]];
                    vec3f v02 = pos[pointsIndice[2]] - pos[pointsIndice[1]];
                    vec3f new_pt = pos[pointsIndice[0]] + v01 * r1 + v02 * r2;
                    newPts[i] = new_pt;
                }
                else {
                    //TODO: 三角面
                }
            });
        }
        spOutput->create_point_attr("pos", newPts);
        return spOutput;
    }

    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> constructGeom(const std::vector<std::vector<zeno::vec3f>>& faces) {
        int nPoints = 0, nFaces = faces.size();
        for (auto& facePts : faces) {
            nPoints += facePts.size();
        }
        auto spOutput = create_GeometryObject(zeno::Topo_HalfEdge, false, nPoints, nFaces);
        std::vector<vec3f> points;
        points.resize(nPoints);
        int nInitPoints = 0;
        for (int iFace = 0; iFace < faces.size(); iFace++) {
            const std::vector<zeno::vec3f>& facePts = faces[iFace];
            zeno::Vector<int> ptIndice;
            for (int iPt = 0; iPt < facePts.size(); iPt++) {
                int currIdx = nInitPoints + iPt;
                points[currIdx] = facePts[iPt];
                ptIndice.push_back(currIdx);
            }
            nInitPoints += facePts.size();
            spOutput->add_face(ptIndice);
        }
        spOutput->create_attr(ATTR_POINT, "pos", points);
        return spOutput;
    }

    static void copyAttribute(
        GeometryObject* srcObj,
        GeometryObject* destObj,
        GeoAttrGroup group,
        std::unordered_map<int, int> oldtonew,
        std::set<std::string> excludeNames)
    {
        std::vector<std::string> attrnames = srcObj->get_attr_names(group);
        for (auto attr : attrnames) {
            if (excludeNames.find(attr) != excludeNames.end())
                continue;

            if (!destObj->has_attr(group, attr)) {
                zeno::GeoAttrType type = srcObj->get_attr_type(group, attr);
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
                destObj->create_attr(group, attr, init_val);
            }

            for (int i = 0; i < destObj->get_group_count(group); i++) {
                int newidx = oldtonew[i];
                AttrValue val = srcObj->get_attr_elem(group, attr, i);
                destObj->set_attr_elem(group, attr, newidx, val);
            }
        }
    }

    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> fuseGeometry(zeno::GeometryObject_Adapter* input, float threshold) {

        std::vector<vec3f> points = input->m_impl->points_pos();
        _PointCloud cloud{ points };
        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, _PointCloud>,
            _PointCloud, 3> KDTree;

        KDTree tree(3, cloud, { 10 });
        tree.buildIndex();

        int n = points.size();
        UnionFind uf(n);

        std::vector<size_t> ret_indices(100);
        std::vector<float> out_dist_sqr(100);

        std::vector<nanoflann::ResultItem<uint32_t, float>> ret_matches;
        nanoflann::SearchParameters params;

        for (size_t i = 0; i < n; ++i) {
            size_t count = tree.radiusSearch(&points[i][0], threshold * threshold, ret_matches, params);
            assert(count == ret_matches.size());
            for (size_t j = 0; j < count; ++j) {
                uf.unite(i, ret_matches[j].first);
            }
        }

        // 计算每个合并组的中心点
        std::unordered_map<int, std::vector<int>> groups;
        for (int i = 0; i < n; ++i) {
            groups[uf.find(i)].push_back(i);
        }

        std::vector<vec3f> new_points;
        std::unordered_map<int, int> pointMapping;
        for (const auto& group : groups) {
            vec3f centroid = { 0, 0, 0 };
            for (int idx : group.second) {
                for (int d = 0; d < 3; ++d) {
                    centroid[d] += points[idx][d];
                }
            }
            for (float& c : centroid)
                c /= group.second.size();
            pointMapping[group.first] = new_points.size();
            new_points.push_back(centroid);
        }

        std::vector<std::vector<int>> faces = input->m_impl->face_indice();
        // 更新面数据
        std::vector<std::vector<int>> newFaces;
        for (const auto& face : faces) {
            std::vector<int> newFace;
            for (int v : face) {
                newFace.push_back(pointMapping[uf.find(v)]);
            }
            // 防止生成重复面
            std::vector<int> _face = newFace;
            std::sort(_face.begin(), _face.end());
            _face.erase(std::unique(_face.begin(), _face.end()), _face.end());
            if (_face.size() >= 3) {
                newFaces.push_back(newFace);
            }
            else {
                int j = 0;
            }
        }

        auto spOutput = create_GeometryObject(zeno::Topo_HalfEdge, false, new_points.size(), newFaces.size());
        for (int iFace = 0; iFace < newFaces.size(); iFace++) {
            const std::vector<int>& facePts = newFaces[iFace];
            const zeno::Vector<int>& _facePts = stdVec2zeVec(facePts);
            spOutput->add_face(_facePts);
        }
        spOutput->create_attr(ATTR_POINT, "pos", new_points);
        copyAttribute(input->m_impl.get(), spOutput->m_impl.get(), ATTR_POINT, pointMapping, {"pos"});
        if (newFaces.size() == faces.size()) {
            //just a patch...
            spOutput->inheritAttributes(input, -1, -1, {}, 0, {});
        }
        return spOutput;
    }

    ZENO_API glm::mat4 calc_rotate_matrix(
        float xangle,
        float yangle,
        float zangle,
        Rotate_Orientaion orientaion
    ) {
        float rad_x = xangle * (M_PI / 180.0);
        float rad_y = yangle * (M_PI / 180.0);
        float rad_z = zangle * (M_PI / 180.0);
#if 0
        switch (orientaion)
        {
        case Orientaion_XY: //绕x轴旋转90
            rad_x = (xangle + 90) * (M_PI / 180.0);
            break;
        case Orientaion_YZ: //绕z轴旋转-90
            rad_z = (zangle - 90) * (M_PI / 180.0);
            break;
        case Orientaion_ZX:
            break;//默认都是基于ZX平面
        }
#endif
        //这里构造的方式是基于列，和公式上的一样，所以看起来反过来了
        glm::mat4 mx = glm::mat4(
            1, 0, 0, 0,
            0, cos(rad_x), sin(rad_x), 0,
            0, -sin(rad_x), cos(rad_x), 0,
            0, 0, 0, 1);
        glm::mat4 my = glm::mat4(
            cos(rad_y), 0, -sin(rad_y), 0,
            0, 1, 0, 0,
            sin(rad_y), 0, cos(rad_y), 0,
            0, 0, 0, 1);
        glm::mat4 mz = glm::mat4(
            cos(rad_z), sin(rad_z), 0, 0,
            -sin(rad_z), cos(rad_z), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
        return mz * my * mx;
    }
}