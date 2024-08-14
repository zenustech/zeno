#include "DualQuaternion.h"
#include <cmath>
#include <zeno/utils/bit_operations.h>

DualQuaternion operator+(const DualQuaternion& l, const DualQuaternion& r) {
	return DualQuaternion(l.real + r.real, l.dual + r.dual);
}

DualQuaternion operator*(const DualQuaternion& dq, float f) {
	return DualQuaternion(dq.real * f, dq.dual * f);
}

bool operator==(const DualQuaternion& l, const DualQuaternion& r) {
	return l.real == r.real && l.dual == r.dual;
}

bool operator!=(const DualQuaternion& l, const DualQuaternion& r) {
	return l.real != r.real || l.dual != r.dual;
}

// Remember, multiplication order is left to right. 
// This is the opposite of matrix and quaternion multiplication order
DualQuaternion operator*(const DualQuaternion& l, const DualQuaternion& r) {
	DualQuaternion lhs = normalized(l);
	DualQuaternion rhs = normalized(r);
//	DualQuaternion lhs = l;
//	DualQuaternion rhs = r;

	return DualQuaternion(lhs.real * rhs.real, lhs.real * rhs.dual + lhs.dual * rhs.real);
}

float dot(const DualQuaternion& l, const DualQuaternion& r) {
	return dot(l.real, r.real);
}

DualQuaternion conjugate(const DualQuaternion& dq) {
	return DualQuaternion(conjugate(dq.real), conjugate(dq.dual));
}

DualQuaternion normalized(const DualQuaternion& dq) {
	float magSq = dot(dq.real, dq.real);
	if (magSq < 0.000001f) {
		return DualQuaternion();
	}
	float invMag = 1.0f / sqrtf(magSq);

	return DualQuaternion(dq.real * invMag, dq.dual * invMag);
}

void normalize(DualQuaternion& dq) {
	float magSq = dot(dq.real, dq.real);
	if (magSq < 0.000001f) {
		return;
	}
	float invMag = 1.0f / sqrtf(magSq);

	dq.real = dq.real * invMag;
	dq.dual = dq.dual * invMag;
}

static void decomposeMtx(const glm::mat4& m, glm::vec3& pos, glm::quat& rot, glm::vec3& scale)
{
    pos = m[3];
    for(int i = 0; i < 3; i++)
        scale[i] = glm::length(glm::vec3(m[i]));
    const glm::mat3 rotMtx(
        glm::vec3(m[0]) / scale[0],
        glm::vec3(m[1]) / scale[1],
        glm::vec3(m[2]) / scale[2]);
    rot = glm::quat_cast(rotMtx);
}

constexpr glm::quat dual_quat(const glm::quat& q,const glm::vec3& t) {

    auto qx = q.x;
    auto qy = q.y;
    auto qz = q.z;
    auto qw = q.w;
    auto tx = t[0];
    auto ty = t[1];
    auto tz = t[2];

    glm::quat qd;
    qd.w = -0.5*( tx*qx + ty*qy + tz*qz);          // qd.w
    qd.x =  0.5*( tx*qw + ty*qz - tz*qy);          // qd.x
    qd.y =  0.5*(-tx*qz + ty*qw + tz*qx);          // qd.y
    qd.z =  0.5*( tx*qy - ty*qx + tz*qw);          // qd.z

    return qd;
}
DualQuaternion mat4ToDualQuat2(const glm::mat4& transformation) {
    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translation;
    decomposeMtx(transformation, translation, rotation, scale);
	glm::quat qr = rotation;
	glm::quat qd = dual_quat(qr, translation);
	return DualQuaternion(qr, qd);
}

glm::mat4 dualQuatToMat4(const DualQuaternion& dq) {
	glm::mat4 rotation = glm::toMat4(dq.real);

	glm::quat d = conjugate(dq.real) * (dq.dual * 2.0f);
	glm::mat4 position = glm::translate(glm::vec3(d.x, d.y, d.z));

	glm::mat4 result = position * rotation;
	return result;
}

glm::vec3 transformVector(const DualQuaternion& dq, const glm::vec3& v) {
	return dq.real * v;
}

glm::vec3 transformPoint2(const DualQuaternion& dq, const glm::vec3& v){
    auto d0 = glm::vec3(dq.real.x, dq.real.y, dq.real.z);
    auto de = glm::vec3(dq.dual.x, dq.dual.y, dq.dual.z);
    auto a0 = dq.real.w;
    auto ae = dq.dual.w;

    return v + 2.0f * cross(d0,cross(d0,v) + a0*v) + 2.0f *(a0*de - ae*d0 + cross(d0,de));
}


zeno::vec3f transformVector(const DualQuaternion& dq, const zeno::vec3f& v) {
    return zeno::bit_cast<zeno::vec3f>(transformVector(dq, zeno::bit_cast<glm::vec3>(v)));
}

zeno::vec3f transformPoint2(const DualQuaternion& dq, const zeno::vec3f& v) {
    return zeno::bit_cast<zeno::vec3f>(transformPoint2(dq, zeno::bit_cast<glm::vec3>(v)));
}