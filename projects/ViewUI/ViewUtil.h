#include <math.h>
#define eps 1e-8
#define zero(x) (((x) > 0 ? (x) : -(x)) < eps)

struct point3 {
	point3() {}
    point3(zeno::vec3f p) {
        x = p[0];
        y = p[1];
        z = p[2];
    }
	point3(float x, float y, float z) {
		x = x;
		y = y;
		z = z;
	}
    double x, y, z;

	double abs() { return sqrt(norm()); }
	double norm() { return x * x + y * y; }
	zeno::vec3f toVec3f() {
		return zeno::vec3f(x, y, z);
	}

	point3 operator + (point3 p) { return point3(x + p.x, y + p.y, z + p.z); }
	point3 operator - (point3 p) { return point3(x - p.x, y - p.y, z - p.z); }
	point3 operator * (double a) { return point3(a * x, a * y, a * z); }
	point3 operator / (double a) { return point3(x / a, y / a, y / z); }
	bool operator == (const point3& p) const {
		return fabs(x - p.x) < eps && fabs(y - p.y) < eps && fabs(z - p.z) < eps;
	}
};
struct line3 {
    point3 a, b;
};
struct plane3 {
    point3 a, b, c;
};

enum CarModelOptionType
{
	Head = 2,
	Tail,
	Top,
	HeadOutline,
	TopProfile,
	SideProfile
};


struct TrisSharedEdge
{
	int tris1 = -1;
	int tris2 = -1;
	zeno::vec2i edge;
	zeno::vec2i previous;
	zeno::vec2i next;
};

struct TriangleData
{
	int idx;
	zeno::vec3i pntIdx;
};

//字符串分割
std::vector<std::string> split(const std::string& str, char delim) {
	std::stringstream ss(str);
	std::vector<std::string> tokens;
	std::string token;

	while (std::getline(ss, token, delim)) {
		if (!token.empty())
			tokens.push_back(token);
	}
	return tokens;
}

std::vector<std::vector<double>> separateSimilarValues(const std::vector<double>& lst, double threshold) {
	std::vector<std::vector<double>> separatedList;
	std::vector<double> currentGroup;
	currentGroup.push_back(lst[0]);

	for (int i = 1; i < lst.size(); ++i) {
		if (std::abs(lst[i] - lst[i - 1]) <= threshold) {
			currentGroup.push_back(lst[i]);
		}
		else {
			separatedList.push_back(currentGroup);
			currentGroup.clear();
			currentGroup.push_back(lst[i]);
		}
	}

	separatedList.push_back(currentGroup);
	return separatedList;
}

//计算两点之间的向量
zeno::vec3f calVector(zeno::vec3f p1, zeno::vec3f p2) {
	zeno::vec3f p;
	p[0] = p1[0] - p2[0];
	p[1] = p1[1] - p2[1];
	p[2] = p1[2] - p2[2];
	return p;
}
//计算两个向量的交叉积
zeno::vec3f calCross(zeno::vec3f p1, zeno::vec3f p2) {
	zeno::vec3f p;
	p[0] = p1[2] * p2[1] - p1[1] * p2[2];
	p[1] = p1[0] * p2[2] - p1[2] * p2[0];
	p[2] = p1[1] * p2[0] - p1[0] * p2[1];
	return p;
}
// 点积
float calDot(zeno::vec3f p1, zeno::vec3f p2) {
	return p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2];
}

//计算向量长度
float calMag(zeno::vec3f pv) {
	return sqrt(pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
}
//向量单位化
zeno::vec3f calNormalize(zeno::vec3f pv) {
	zeno::vec3f p = pv;
	float Magnitude = calMag(p); // 获得矢量的长度
	p[0] /= Magnitude;
	p[1] /= Magnitude;
	p[2] /= Magnitude;
	return p;
}
//计算三角形法线
zeno::vec3f calculateTriNormal(zeno::vec3f p1, zeno::vec3f p2, zeno::vec3f p3) {
	zeno::vec3f triNormal = calCross(calVector(p1, p3), calVector(p3, p2));
	return normalize(triNormal);
}
// 计算两点距离
double calPointDistance(zeno::vec3f p1, zeno::vec3f p2) {
	return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
		(p1[1] - p2[1]) * (p1[1] - p2[1]) +
		(p1[2] - p2[2]) * (p1[2] - p2[2]));
}
//根据三个点计算新点坐标
zeno::vec3f calNewPointPosition(zeno::vec3f p1, zeno::vec3f p2, zeno::vec3f p3) {
	zeno::vec3f p;
	p[0] = p1[0] + p2[0] - p3[0];
	p[1] = p1[1] + p2[1] - p3[1];
	p[2] = p1[2] + p2[2] - p3[2];
	return p;
}
//计算三角形面积
float calculateArea(zeno::vec3f p1, zeno::vec3f p2, zeno::vec3f p3) {
	float len1 = distance(p1, p2);//calPointDistance
	float len2 = distance(p1, p3);
	float len3 = distance(p3, p2);
	float s = 0.5 * (len1 + len2 + len3);
	return sqrt(s * (s - len1) * (s - len2) * (s - len3));
}
// 判断两个向量v1和v2是否指向同一方向
bool calSameSide(zeno::vec3f A, zeno::vec3f B, zeno::vec3f C, zeno::vec3f P) {
	zeno::vec3f AB = A - P;
	zeno::vec3f AC = B - P;
	zeno::vec3f AP = C - P;

	zeno::vec3f v1 = calCross(AB, AC);
	zeno::vec3f v2 = calCross(AB, AP);

	return calDot(v1, v2) >= 0;
}
//判断点是否在三角形内
bool calculatePntInTril(zeno::vec3f A, zeno::vec3f B, zeno::vec3f C, zeno::vec3f P) {
	return calSameSide(A, B, C, P) && calSameSide(B, C, A, P) && calSameSide(C, A, B, P);
}
// 判断点是否在三角形内 重心法
bool calculatePntInTrilEx(zeno::vec3f A, zeno::vec3f B, zeno::vec3f C, zeno::vec3f P) {
	zeno::vec3f v0 = A - P;
	zeno::vec3f v1 = B - P;
	zeno::vec3f v2 = C - P;

	float dot00 = calDot(v0, v0);
	float dot01 = calDot(v0, v1);
	float dot02 = calDot(v0, v2);
	float dot11 = calDot(v1, v1);
	float dot12 = calDot(v1, v2);

	float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

	float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
	if (u < 0 || u > 1) // if u out of range, return directly
	{
		return false;
	}

	float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
	if (v < 0 || v > 1) // if v out of range, return directly
	{
		return false;
	}

	return u + v <= 1;
}
#if 0
// 判断点是否在平面内
bool isPointInPlane(zeno::vec3f A, zeno::vec3f B, zeno::vec3f C, zeno::vec3f P) {
	// 计算点到平面的向量
	zeno::vec3f vectorToPoint;
	vectorToPoint[0] = P[0] - plane.p[0];
	vectorToPoint[1] = P[1] - plane.p[1];
	vectorToPoint[2] = P[2] - plane.p[2];

	auto normal = calculateTriNormal(A, B, C);
	// 计算点到平面的距离（点积）
	float distance = vectorToPoint.x * normal.x +
		vectorToPoint.y * normal.y +
		vectorToPoint.z * normal.z;

	if (distance == 0) {
		return true;  // 点在平面上
	}
	else {
		return false;  // 点不在平面上
	}
}
#endif
/*
@description 判断点point是否在由顶点数组vertices所指定的多边形内部
思想：将点point和多边形所有的顶点链接起来，计算相邻两边的夹角之和，
若和等于360°，那说明该点就在多边形内。
参考链接：http://www.html-js.com/article/1538
@param  point 待判断的点。有两个分量的List。
@param  vertices 多边形顶点数组，其中的前后相邻的元素在多边形上也
是相邻的。3个以上的二分量List(一个二分量List为一个顶点)组成的List。
@return 若在多边形之内或者在多边形的边界上，返回True，否则返回False
*/
bool is_in_2d_polygon(zeno::vec2f point, std::vector<zeno::vec2f> vertices)
{
	float px = point[0];
	float py = point[1];
	float angle_sum = 0;

	int j = vertices.size() - 1;
	for (int i = 0; i < vertices.size(); i++) {
		float sx = vertices[i][0];
		float sy = vertices[i][1];
		float tx = vertices[j][0];
		float ty = vertices[j][1];

		// 通过判断点到通过两点的直线的距离是否为0来判断点是否在边上
		// y = kx + b, -y + kx + b = 0
		float k = (sy - ty) / (sx - tx + 0.000000000001);  // 避免除0
		float b = sy - k * sx;
		float dis = fabs(k * px - 1 * py + b) / sqrt(k * k + 1);
		if (dis < 0.000001) {  // 该点在直线上
			if (sx <= px <= tx || tx <= px <= sx)  // 该点在边的两个定点之间，说明顶点在边上
				return true;
		}

		// 计算夹角
		float angle = atan2(sy - py, sx - px) - atan2(ty - py, tx - px);
		// angle需要在-π到π之内
		if (angle >= M_PI)
			angle = angle - M_PI * 2;
		else if (angle <= -M_PI)
			angle = angle + M_PI * 2;

		// 累积
		angle_sum += angle;
		j = i;
	}

	//计算夹角和于2* pi之差，若小于一个非常小的数，就认为相等
	return fabs(angle_sum - M_PI * 2) < 0.00000000001;
}

//@description 判断空间点point是否在顶点集为vertices，法向量为normal的多边形内
//@param point List(3) 三分量List，表示一个三维点。
//@param normal List(3) 三分量List，表示一个三维向量。
//@param vertices List(n > 3)[List(3)] 由n个三维点组成的List。
//@return 若在多边形之内或者在多边形的边界上，返回True，否则返回False。
bool is_in_3d_polygon(std::vector<zeno::vec3f> tri, zeno::vec3f P) {
	//复制数据
	std::vector<zeno::vec3f> local_v = tri;
	zeno::vec3f local_p = P;
	auto normal = calculateTriNormal(local_v[0], local_v[1], local_v[2]);
	// 求点到平面的距离, 点到平面的距离公式
	auto na = normal[0];
	auto nb = normal[1];
	auto nc = normal[2];
	auto d = -(na * local_v[0][0] + nb * local_v[0][1] + nc * local_v[0][2]);
	auto distance = fabs(na * local_p[0] + nb * local_p[1] + nc * local_p[2] + d) \
		/ (sqrt(na * na + nb * nb + nc * nc));
	//点不在平面上，肯定不在多边形内
	if (distance > 0) return false;

	int index = 2;  // 默认删除z分量
	if (normal[0] != 0)index = 0;  // 删除x分量
	else if (normal[1] != 0)index = 1; //删除y分量
	else if (normal[2] != 0)index = 2; //删除z分量
	//删除P和顶点集中指定的分量
	zeno::vec2f tmpLocal_p;
	if (index == 0) tmpLocal_p = { local_p[1],local_p[2] };
	if (index == 1) tmpLocal_p = { local_p[0],local_p[2] };
	if (index == 2) tmpLocal_p = { local_p[0],local_p[1] };

	std::vector<zeno::vec2f> tmpLocal_v;
	for (auto p : local_v) {
		if (index == 0) tmpLocal_v.push_back({ p[1],p[2] });
		if (index == 1) tmpLocal_v.push_back({ p[0],p[2] });
		if (index == 2) tmpLocal_v.push_back({ p[0],p[1] });
	}
	//调用二维的判断点是否在多边形内的方法。
	return is_in_2d_polygon(tmpLocal_p, tmpLocal_v);
}

// 查找三角形共同边
zeno::vec2i findCommonEdges(std::vector<zeno::vec2i> triangle1, std::vector<zeno::vec2i> triangle2) {
	for (const auto& edge1 : triangle1) {
		for (const auto& edge2 : triangle2) {
			if ((edge1[0] == edge2[0] && edge1[1] == edge2[1]) ||
				(edge1[0] == edge2[1] && edge1[1] == edge2[0])) {
				return edge1;
			}
		}
	}
	return {};
}

std::vector<zeno::vec3i> addtri(zeno::vec3i tri,
	std::vector<int> preEdge,
	std::vector<int> currentEdge,
	std::vector<int> nextEdge,
	int currentPntidx, int prePntidx, int nextPntidx)
{
	std::vector<zeno::vec3i> tmpTris;
	zeno::vec3i tmpTri = tri;
	//添加三角形
	int v0 = tmpTri[0];
	int v1 = tmpTri[1];
	int v2 = tmpTri[2];
	if (v0 < v1) std::swap(v0, v1);
	if (v0 < v2) std::swap(v0, v2);
	if (v1 < v2) std::swap(v1, v2);

	//中间的
	if (preEdge.size() > 0 && nextEdge.size() > 0)
	{
		auto tv10 = currentEdge[0];
		auto tv11 = currentEdge[1];

		//tmpTris.push_back({ tv10, prePntidx, currentPntidx });
		//tmpTris.push_back({ tv11, currentPntidx, prePntidx });
		tmpTris.push_back({ tv10, currentPntidx, prePntidx });
		tmpTris.push_back({ prePntidx, currentPntidx, tv11 });

		auto tv20 = nextEdge[0];
		auto tv21 = nextEdge[1];
		if (tv10 == tv20 && tv11 != tv21)
		{
			tmpTris.push_back({ tv21, tv11, currentPntidx });
		}
		else if (tv10 == tv21 && tv11 != tv20)
		{
			tmpTris.push_back({ tv11, tv20, currentPntidx });
		}
		else
		{
			if (tv11 == tv20)
			{
				tmpTris.push_back({ tv21, tv10, currentPntidx });
			}
			else
			{
				tmpTris.push_back({ tv10, tv20, currentPntidx });
			}
		}
	}
	//开始的
	else if (preEdge.size() == 0 && nextEdge.size() > 0)
	{
		auto tv0 = currentEdge[0];
		auto tv1 = currentEdge[1];

		if ((tv0 == v0 && tv1 == v1) || (tv0 == v1 && tv1 == v0))
		{
			tmpTris.push_back({ v1, v2, currentPntidx });
			tmpTris.push_back({ v2, v0, currentPntidx });
		}
		if ((tv0 == v0 && tv1 == v2) || (tv0 == v2 && tv1 == v0))
		{
			tmpTris.push_back({ v0, v1, currentPntidx });
			tmpTris.push_back({ v1, v2, currentPntidx });
		}
		if ((tv0 == v1 && tv1 == v2) || (tv0 == v2 && tv1 == v1))
		{
			tmpTris.push_back({ v0, v1, currentPntidx });
			tmpTris.push_back({ v2, v0, currentPntidx });
		}
		tmpTris.push_back({ tv0, nextPntidx, currentPntidx });
		tmpTris.push_back({ currentPntidx, nextPntidx, tv1 });
	}
	//结束的
	else if (preEdge.size() > 0 && nextEdge.size() == 0)
	{
		auto tv0 = currentEdge[0];
		auto tv1 = currentEdge[1];
		tmpTris.push_back({ tv0, currentPntidx, prePntidx });
		tmpTris.push_back({ prePntidx, currentPntidx, tv1 });

		if ((tv0 == v0 && tv1 == v1) || (tv0 == v1 && tv1 == v0))
		{
			tmpTris.push_back({ v0, v2, currentPntidx });
			tmpTris.push_back({ v1, v2, currentPntidx });
		}
		if ((tv0 == v0 && tv1 == v2) || (tv0 == v2 && tv1 == v0))
		{
			tmpTris.push_back({ v0, v1, currentPntidx });
			tmpTris.push_back({ v1, v2, currentPntidx });
		}
		if ((tv0 == v1 && tv1 == v2) || (tv0 == v2 && tv1 == v1))
		{
			tmpTris.push_back({ v0, v1, currentPntidx });
			tmpTris.push_back({ v2, v0, currentPntidx });
		}
	}
	return tmpTris;
}

std::vector<TrisSharedEdge> findSharedEdgeTriOrd(std::vector<TriangleData> tmpDTris)
{
	std::vector<TrisSharedEdge> tmpDTrisEdges;
	std::vector<zeno::vec2i> triangle1, triangle2;
	int CommonEdgeTriIdx1, CommonEdgeTriIdx2;
	for (auto iiter1 = tmpDTris.begin(); iiter1 != tmpDTris.end(); iiter1++)
	{
		for (auto iiter2 = tmpDTris.begin(); iiter2 != tmpDTris.end(); iiter2++)
		{
			bool bExist = false;
			for (auto tes : tmpDTrisEdges)
			{
				if (iiter2->idx == tes.tris1 || iiter2->idx == tes.tris2)
				{
					bExist = true;
					break;
				}
			}
			if (bExist == true)
			{
				continue;
			}
			int v0 = iiter2->pntIdx[0];
			int v1 = iiter2->pntIdx[1];
			int v2 = iiter2->pntIdx[2];
			if (v0 < v1) std::swap(v0, v1);
			if (v0 < v2) std::swap(v0, v2);
			if (v1 < v2) std::swap(v1, v2);

			if (iiter1 == iiter2)
			{
				if (triangle1.size() == 0)
				{
					CommonEdgeTriIdx1 = iiter1->idx;
					triangle1.push_back({ v0, v1 });
					triangle1.push_back({ v1, v2 });
					triangle1.push_back({ v0, v2 });
				}
				continue;
			}
			if (triangle2.size() == 0)
			{
				CommonEdgeTriIdx2 = iiter2->idx;
				triangle2.push_back({ v0, v1 });
				triangle2.push_back({ v1, v2 });
				triangle2.push_back({ v0, v2 });
			}
			auto tmpEdge = findCommonEdges(triangle1, triangle2);
			if (tmpEdge[0] == tmpEdge[1])
			{
				triangle2.clear();
				continue;
			}
			else
			{
				TrisSharedEdge tmptrisedge;
				tmptrisedge.tris1 = CommonEdgeTriIdx1;
				tmptrisedge.tris2 = CommonEdgeTriIdx2;
				tmptrisedge.edge = tmpEdge;
				if (tmpDTrisEdges.size() > 0)
				{
					tmpDTrisEdges.back().next = tmptrisedge.edge;
					tmptrisedge.previous = tmpDTrisEdges.back().edge;
				}
				tmpDTrisEdges.push_back(tmptrisedge);

				CommonEdgeTriIdx1 = CommonEdgeTriIdx2;
				triangle1 = triangle2;
				triangle2.clear();
				break;
			}
		}
	}
	return tmpDTrisEdges;
}

//计算cross product U x V
point3 xmult(point3 u, point3 v) {
    point3 ret;
    ret.x = u.y * v.z - v.y * u.z;
    ret.y = u.z * v.x - u.x * v.z;
    ret.z = u.x * v.y - u.y * v.x;
    return ret;
}

//计算dot product U . V
double dmult(point3 u, point3 v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

//矢量差 U - V
point3 subt(point3 u, point3 v) {
    point3 ret;
    ret.x = u.x - v.x;
    ret.y = u.y - v.y;
    ret.z = u.z - v.z;
    return ret;
}

//取平面法向量
point3 pvec(plane3 s) {
    return xmult(subt(s.a, s.b), subt(s.b, s.c));
}
point3 pvec(point3 s1, point3 s2, point3 s3) {
    return xmult(subt(s1, s2), subt(s2, s3));
}

//两点距离,单参数取向量大小
double distance(point3 p1, point3 p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

//向量大小
double vlen(point3 p) {
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

//判三点共线
int dots_inline(point3 p1, point3 p2, point3 p3) {
    return vlen(xmult(subt(p1, p2), subt(p2, p3))) < eps;
}

//判四点共面
int dots_onplane(point3 a, point3 b, point3 c, point3 d) {
    return zero(dmult(pvec(a, b, c), subt(d, a)));
}

//判点是否在线段上,包括端点和共线
int dot_online_in(point3 p, line3 l) {
    return zero(vlen(xmult(subt(p, l.a), subt(p, l.b)))) && (l.a.x - p.x) * (l.b.x - p.x) < eps &&
           (l.a.y - p.y) * (l.b.y - p.y) < eps && (l.a.z - p.z) * (l.b.z - p.z) < eps;
}
int dot_online_in(point3 p, point3 l1, point3 l2) {
    return zero(vlen(xmult(subt(p, l1), subt(p, l2)))) && (l1.x - p.x)*(l2.x - p.x) < eps &&
           (l1.y - p.y) * (l2.y - p.y) < eps && (l1.z - p.z) * (l2.z - p.z) < eps;
}

//判点是否在线段上,不包括端点
int dot_online_ex(point3 p, line3 l) {
    return dot_online_in(p, l) && (!zero(p.x - l.a.x) || !zero(p.y - l.a.y) || !zero(p.z - l.a.z)) &&
           (!zero(p.x - l.b.x) || !zero(p.y - l.b.y) || !zero(p.z - l.b.z));
}
int dot_online_ex(point3 p, point3 l1, point3 l2) {
    return dot_online_in(p, l1, l2) && (!zero(p.x - l1.x) || !zero(p.y - l1.y) || !zero(p.z - l1.z)) &&
           (!zero(p.x - l2.x) || !zero(p.y - l2.y) || !zero(p.z - l2.z));
}

//判点是否在空间三角形上,包括边界,三点共线无意义
int dot_inplane_in(point3 p, plane3 s) {
    return zero(vlen(xmult(subt(s.a, s.b), subt(s.a, s.c))) - vlen(xmult(subt(p, s.a), subt(p, s.b))) -
                vlen(xmult(subt(p, s.b), subt(p, s.c))) - vlen(xmult(subt(p, s.c), subt(p, s.a))));
}
int dot_inplane_in(point3 p, point3 s1, point3 s2, point3 s3) {
    return zero(vlen(xmult(subt(s1, s2), subt(s1, s3))) - vlen(xmult(subt(p, s1), subt(p, s2))) -
                vlen(xmult(subt(p, s2), subt(p, s3))) - vlen(xmult(subt(p, s3), subt(p, s1))));
}

//判点是否在空间三角形上,不包括边界,三点共线无意义
int dot_inplane_ex(point3 p, plane3 s) {
    return dot_inplane_in(p, s) && vlen(xmult(subt(p, s.a), subt(p, s.b))) > eps &&
           vlen(xmult(subt(p, s.b), subt(p, s.c))) > eps && vlen(xmult(subt(p, s.c), subt(p, s.a))) > eps;
}
int dot_inplane_ex(point3 p, point3 s1, point3 s2, point3 s3) {
    return dot_inplane_in(p, s1, s2, s3) && vlen(xmult(subt(p, s1), subt(p, s2))) > eps &&
           vlen(xmult(subt(p, s2), subt(p, s3))) > eps && vlen(xmult(subt(p, s3), subt(p, s1))) > eps;
}

//判两点在线段同侧,点在线段上返回0,不共面无意义
int same_side(point3 p1, point3 p2, line3 l) {
    return dmult(xmult(subt(l.a, l.b), subt(p1, l.b)), xmult(subt(l.a, l.b), subt(p2, l.b))) > eps;
}
int same_side(point3 p1, point3 p2, point3 l1, point3 l2) {
    return dmult(xmult(subt(l1, l2), subt(p1, l2)), xmult(subt(l1, l2), subt(p2, l2))) > eps;
}

//判两点在线段异侧,点在线段上返回0,不共面无意义
int opposite_side(point3 p1, point3 p2, line3 l) {
    return dmult(xmult(subt(l.a, l.b), subt(p1, l.b)), xmult(subt(l.a, l.b), subt(p2, l.b))) < -eps;
}
int opposite_side(point3 p1, point3 p2, point3 l1, point3 l2) {
    return dmult(xmult(subt(l1, l2), subt(p1, l2)), xmult(subt(l1, l2), subt(p2, l2))) < -eps;
}

//判两点在平面同侧,点在平面上返回0
int same_side(point3 p1, point3 p2, plane3 s) {
    return dmult(pvec(s), subt(p1, s.a)) * dmult(pvec(s), subt(p2, s.a)) > eps;
}
int same_side(point3 p1, point3 p2, point3 s1, point3 s2, point3 s3) {
    return dmult(pvec(s1, s2, s3), subt(p1, s1)) * dmult(pvec(s1, s2, s3), subt(p2, s1)) > eps;
}

//判两点在平面异侧,点在平面上返回0
int opposite_side(point3 p1, point3 p2, plane3 s) {
    return dmult(pvec(s), subt(p1, s.a)) * dmult(pvec(s), subt(p2, s.a)) < -eps;
}
int opposite_side(point3 p1, point3 p2, point3 s1, point3 s2, point3 s3) {
    return dmult(pvec(s1, s2, s3), subt(p1, s1)) * dmult(pvec(s1, s2, s3), subt(p2, s1)) < -eps;
}

//判两直线平行
int parallel(line3 u, line3 v) {
    return vlen(xmult(subt(u.a, u.b), subt(v.a, v.b))) < eps;
}
int parallel(point3 u1, point3 u2, point3 v1, point3 v2) {
    return vlen(xmult(subt(u1, u2), subt(v1, v2))) < eps;
}

//判两平面平行
int parallel(plane3 u, plane3 v) {
    return vlen(xmult(pvec(u), pvec(v))) < eps;
}
int parallel(point3 u1, point3 u2, point3 u3, point3 v1, point3 v2, point3 v3) {
    return vlen(xmult(pvec(u1, u2, u3), pvec(v1, v2, v3))) < eps;
}

//判直线与平面平行
int parallel(line3 l, plane3 s) {
    return zero(dmult(subt(l.a, l.b), pvec(s)));
}
int parallel(point3 l1, point3 l2, point3 s1, point3 s2, point3 s3) {
    return zero(dmult(subt(l1, l2), pvec(s1, s2, s3)));
}

//判两直线垂直
int perpendicular(line3 u, line3 v) {
    return zero(dmult(subt(u.a, u.b), subt(v.a, v.b)));
}
int perpendicular(point3 u1, point3 u2, point3 v1, point3 v2) {
    return zero(dmult(subt(u1, u2), subt(v1, v2)));
}

//判两平面垂直
int perpendicular(plane3 u, plane3 v) {
    return zero(dmult(pvec(u), pvec(v)));
}
int perpendicular(point3 u1, point3 u2, point3 u3, point3 v1, point3 v2, point3 v3) {
    return zero(dmult(pvec(u1, u2, u3), pvec(v1, v2, v3)));
}

//判直线与平面平行
int perpendicular(line3 l, plane3 s) {
    return vlen(xmult(subt(l.a, l.b), pvec(s))) < eps;
}
int perpendicular(point3 l1, point3 l2, point3 s1, point3 s2, point3 s3) {
    return vlen(xmult(subt(l1, l2), pvec(s1, s2, s3))) < eps;
}

//判两线段相交,包括端点和部分重合
int intersect_in(line3 u, line3 v) {
    if (!dots_onplane(u.a, u.b, v.a, v.b))
        return 0;
    if (!dots_inline(u.a, u.b, v.a) || !dots_inline(u.a, u.b, v.b))
        return !same_side(u.a, u.b, v) && !same_side(v.a, v.b, u);
    return dot_online_in(u.a, v) || dot_online_in(u.b, v) || dot_online_in(v.a, u) || dot_online_in(v.b, u);
}
int intersect_in(point3 u1, point3 u2, point3 v1, point3 v2) {
    if (!dots_onplane(u1, u2, v1, v2))
        return 0;
    if (!dots_inline(u1, u2, v1) || !dots_inline(u1, u2, v2))
        return !same_side(u1, u2, v1, v2) && !same_side(v1, v2, u1, u2);
    return dot_online_in(u1, v1, v2) || dot_online_in(u2, v1, v2) || dot_online_in(v1, u1, u2) ||
           dot_online_in(v2, u1, u2);
}

//判两线段相交,不包括端点和部分重合
int intersect_ex(line3 u, line3 v) {
    return dots_onplane(u.a, u.b, v.a, v.b) && opposite_side(u.a, u.b, v) && opposite_side(v.a, v.b, u);
}
int intersect_ex(point3 u1, point3 u2, point3 v1, point3 v2) {
    return dots_onplane(u1, u2, v1, v2) && opposite_side(u1, u2, v1, v2) && opposite_side(v1, v2, u1, u2);
}

//判线段与空间三角形相交,包括交于边界和(部分)包含
int intersect_in(line3 l, plane3 s) {
    return !same_side(l.a, l.b, s) && !same_side(s.a, s.b, l.a, l.b, s.c) && !same_side(s.b, s.c, l.a, l.b, s.a) &&
           !same_side(s.c, s.a, l.a, l.b, s.b);
}
int intersect_in(point3 l1, point3 l2, point3 s1, point3 s2, point3 s3) {
    return !same_side(l1, l2, s1, s2, s3) && !same_side(s1, s2, l1, l2, s3) && !same_side(s2, s3, l1, l2, s1) &&
           !same_side(s3, s1, l1, l2, s2);
}

//判线段与空间三角形相交,不包括交于边界和(部分)包含
int intersect_ex(line3 l, plane3 s) {
    return opposite_side(l.a, l.b, s) && opposite_side(s.a, s.b, l.a, l.b, s.c) &&
           opposite_side(s.b, s.c, l.a, l.b, s.a) && opposite_side(s.c, s.a, l.a, l.b, s.b);
}
int intersect_ex(point3 l1, point3 l2, point3 s1, point3 s2, point3 s3) {
    return opposite_side(l1, l2, s1, s2, s3) && opposite_side(s1, s2, l1, l2, s3) &&
           opposite_side(s2, s3, l1, l2, s1) && opposite_side(s3, s1, l1, l2, s2);
}

//计算两直线交点,注意事先判断直线是否共面和平行!
//线段交点请另外判线段相交(同时还是要判断是否平行!)
point3 intersection(line3 u, line3 v) {
    point3 ret = u.a;
    double t = ((u.a.x - v.a.x) * (v.a.y - v.b.y) - (u.a.y - v.a.y) * (v.a.x - v.b.x)) /
               ((u.a.x - u.b.x) * (v.a.y - v.b.y) - (u.a.y - u.b.y) * (v.a.x - v.b.x));
    ret.x += (u.b.x - u.a.x) * t;
    ret.y += (u.b.y - u.a.y) * t;
    ret.z += (u.b.z - u.a.z) * t;
    return ret;
}
point3 intersection(point3 u1, point3 u2, point3 v1, point3 v2) {
    point3 ret = u1;
    double t = ((u1.x - v1.x) * (v1.y - v2.y) - (u1.y - v1.y) * (v1.x - v2.x)) /
               ((u1.x - u2.x) * (v1.y - v2.y) - (u1.y - u2.y) * (v1.x - v2.x));
    ret.x += (u2.x - u1.x) * t;
    ret.y += (u2.y - u1.y) * t;
    ret.z += (u2.z - u1.z) * t;
    return ret;
}

//计算直线与平面交点,注意事先判断是否平行,并保证三点不共线!
//线段和空间三角形交点请另外判断
point3 intersection(line3 l, plane3 s) {
    point3 ret = pvec(s);
    double t = (ret.x * (s.a.x - l.a.x) + ret.y * (s.a.y - l.a.y) + ret.z * (s.a.z - l.a.z)) /
               (ret.x * (l.b.x - l.a.x) + ret.y * (l.b.y - l.a.y) + ret.z * (l.b.z - l.a.z));
    ret.x = l.a.x + (l.b.x - l.a.x) * t;
    ret.y = l.a.y + (l.b.y - l.a.y) * t;
    ret.z = l.a.z + (l.b.z - l.a.z) * t;
    return ret;
}
point3 intersection(point3 l1, point3 l2, point3 s1, point3 s2, point3 s3) {
    point3 ret = pvec(s1, s2, s3);
    double t = (ret.x * (s1.x - l1.x) + ret.y * (s1.y - l1.y) + ret.z * (s1.z - l1.z)) /
               (ret.x * (l2.x - l1.x) + ret.y * (l2.y - l1.y) + ret.z * (l2.z - l1.z));
    ret.x = l1.x + (l2.x - l1.x) * t;
    ret.y = l1.y + (l2.y - l1.y) * t;
    ret.z = l1.z + (l2.z - l1.z) * t;
    return ret;
}

//计算两平面交线,注意事先判断是否平行,并保证三点不共线!
line3 intersection(plane3 u, plane3 v) {
    line3 ret;
    ret.a = parallel(v.a, v.b, u.a, u.b, u.c) ? intersection(v.b, v.c, u.a, u.b, u.c)
                                              : intersection(v.a, v.b, u.a, u.b, u.c);
    ret.b = parallel(v.c, v.a, u.a, u.b, u.c) ? intersection(v.b, v.c, u.a, u.b, u.c)
                                              : intersection(v.c, v.a, u.a, u.b, u.c);
    return ret;
}
line3 intersection(point3 u1, point3 u2, point3 u3, point3 v1, point3 v2, point3 v3) {
    line3 ret;
    ret.a = parallel(v1, v2, u1, u2, u3) ? intersection(v2, v3, u1, u2, u3) : intersection(v1, v2, u1, u2, u3);
    ret.b = parallel(v3, v1, u1, u2, u3) ? intersection(v2, v3, u1, u2, u3) : intersection(v3, v1, u1, u2, u3);
    return ret;
}

//点到直线距离
double ptoline(point3 p, line3 l) {
    return vlen(xmult(subt(p, l.a), subt(l.b, l.a))) / distance(l.a, l.b);
}
double ptoline(point3 p, point3 l1, point3 l2) {
    return vlen(xmult(subt(p, l1), subt(l2, l1))) / distance(l1, l2);
}

//点到平面距离
double ptoplane(point3 p, plane3 s) {
    return fabs(dmult(pvec(s), subt(p, s.a))) / vlen(pvec(s));
}
double ptoplane(point3 p, point3 s1, point3 s2, point3 s3) {
    return fabs(dmult(pvec(s1, s2, s3), subt(p, s1))) / vlen(pvec(s1, s2, s3));
}

//直线到直线距离
double linetoline(line3 u, line3 v) {
    point3 n = xmult(subt(u.a, u.b), subt(v.a, v.b));
    return fabs(dmult(subt(u.a, v.a), n)) / vlen(n);
}
double linetoline(point3 u1, point3 u2, point3 v1, point3 v2) {
    point3 n = xmult(subt(u1, u2), subt(v1, v2));
    return fabs(dmult(subt(u1, v1), n)) / vlen(n);
}

//两直线夹角cos值
double angle_cos(line3 u, line3 v) {
    return dmult(subt(u.a, u.b), subt(v.a, v.b)) / vlen(subt(u.a, u.b)) / vlen(subt(v.a, v.b));
}
double angle_cos(point3 u1, point3 u2, point3 v1, point3 v2) {
    return dmult(subt(u1, u2), subt(v1, v2)) / vlen(subt(u1, u2)) / vlen(subt(v1, v2));
}

//两平面夹角cos值
double angle_cos(plane3 u, plane3 v) {
    return dmult(pvec(u), pvec(v)) / vlen(pvec(u)) / vlen(pvec(v));
}
double angle_cos(point3 u1, point3 u2, point3 u3, point3 v1, point3 v2, point3 v3) {
    return dmult(pvec(u1, u2, u3), pvec(v1, v2, v3)) / vlen(pvec(u1, u2, u3)) / vlen(pvec(v1, v2, v3));
}

//直线平面夹角sin值
double angle_sin(line3 l, plane3 s) {
    return dmult(subt(l.a, l.b), pvec(s)) / vlen(subt(l.a, l.b)) / vlen(pvec(s));
}
double angle_sin(point3 l1, point3 l2, point3 s1, point3 s2, point3 s3) {
    return dmult(subt(l1, l2), pvec(s1, s2, s3)) / vlen(subt(l1, l2)) / vlen(pvec(s1, s2, s3));
}