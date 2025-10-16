#pragma once
#include "openvdb/openvdb.h"
#include "zeno/core/IObject.h"

//separately store three channels, used for handling velocity with extrapolation
struct packed_FloatGrid3 : zeno::IObject {
	packed_FloatGrid3() {
		for (int i = 0; i < 3; i++) {
			v[i] = openvdb::FloatGrid::create(0.f);
			v[i]->setName("vc" + std::to_string(i));
		}
		m_transform = openvdb::math::Transform::createLinearTransform(1.0f);
		m_gridclass = openvdb::GridClass::GRID_STAGGERED;
	}
	zeno::zany clone() const override {
		return nullptr;
	}
	~packed_FloatGrid3() = default;
	void setName(std::string name);
	void from_vec3(openvdb::Vec3fGrid::Ptr in_v, bool topologycopy = false);
	void to_vec3(openvdb::Vec3fGrid::Ptr out_v) const;
	void swap(packed_FloatGrid3& other);

	packed_FloatGrid3 deepCopy() const;
	packed_FloatGrid3 fullCopy() const;
	openvdb::FloatGrid::Ptr v[3];
	openvdb::GridClass m_gridclass;
	openvdb::math::Transform::Ptr m_transform;
};
