#ifndef _H_DUALQUATERNION_
#define _H_DUALQUATERNION_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <zeno/utils/vec.h>

struct DualQuaternion {
    glm::quat real = {1, 0, 0, 0};
    glm::quat dual = {0, 0, 0, 0};
	inline DualQuaternion() { }
	inline DualQuaternion(const glm::quat& r, const glm::quat& d) :
		real(r), dual(d) { }
};

DualQuaternion operator+(const DualQuaternion& l, const DualQuaternion& r);
DualQuaternion operator*(const DualQuaternion& dq, float f);
// Multiplication order is left to right
// Left to right is the OPPOSITE of matrices and quaternions
DualQuaternion operator*(const DualQuaternion& l, const DualQuaternion& r);
bool operator==(const DualQuaternion& l, const DualQuaternion& r);
bool operator!=(const DualQuaternion& l, const DualQuaternion& r);

float dot(const DualQuaternion& l, const DualQuaternion& r);
DualQuaternion conjugate(const DualQuaternion& dq);
DualQuaternion normalized(const DualQuaternion& dq);
void normalize(DualQuaternion& dq);

DualQuaternion mat4ToDualQuat2(const glm::mat4& t);
glm::mat4 dualQuatToMat4(const DualQuaternion& dq);

glm::vec3 transformVector(const DualQuaternion& dq, const glm::vec3& v);
zeno::vec3f transformVector(const DualQuaternion& dq, const zeno::vec3f& v);
zeno::vec3f transformPoint2(const DualQuaternion& dq, const zeno::vec3f& v);

#endif