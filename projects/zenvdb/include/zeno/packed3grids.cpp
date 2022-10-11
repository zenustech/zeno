#include "packed3grids.h"
#include "openvdb/tree/LeafManager.h"

void packed_FloatGrid3::setName(std::string name)
{
	for (int i = 0; i < 3; i++) {
		v[i]->setName(name + +"c" + std::to_string(i));
	}
}

void packed_FloatGrid3::from_vec3(openvdb::Vec3fGrid::Ptr in_v, bool topologycopy)
{
	//input is assume to be a staggered grid
	m_transform = in_v->transformPtr()->copy();
	for (int i = 0; i < 3; i++) {
		if (!v[i]) {
			v[i] = openvdb::FloatGrid::create();
		}
		v[i]->setTree(std::make_shared<openvdb::FloatTree>(in_v->tree(), in_v->background()[i], openvdb::TopologyCopy()));
		openvdb::math::Transform::Ptr transform = in_v->transformPtr()->copy();
		if (in_v->getGridClass() == openvdb::GridClass::GRID_STAGGERED) {
			openvdb::Vec3d t{ 0,0,0 };
			t[i] -= 0.5 * in_v->voxelSize()[0];
			transform->postTranslate(t);
		}
		v[i]->setTransform(transform);
		v[i]->setName(in_v->getName() + "c" + std::to_string(i));
	}

	if (topologycopy) {
		return;
	}

	auto converter = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index) {
		//for each channel
		for (int i = 0; i < 3; i++) {
			auto channel_leaf = v[i]->tree().probeLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				channel_leaf->setValueOnly(iter.offset(), iter.getValue()[i]);
			}
		}
	};

	openvdb::tree::LeafManager<openvdb::Vec3fTree> leafman(in_v->tree());
	leafman.foreach(converter);
}


void packed_FloatGrid3::to_vec3(openvdb::Vec3fGrid::Ptr out_v) const
{
	if (!out_v) {
		return;
	}
	out_v->setTree(std::make_shared<openvdb::Vec3fTree>(openvdb::Vec3f(0, 0, 0)));

	for (int i = 0; i < 3; i++) {
		if (!v[i]) {
			return;
		}
		out_v->topologyUnion(*v[i]);
	}

	std::atomic<int> vcounter{ 0 };

	auto filler = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index) {
		for (int i = 0; i < 3; i++) {
			auto channel_leaf = v[i]->tree().probeConstLeaf(leaf.origin());
			if (channel_leaf) {
				for (auto iter = channel_leaf->beginValueOn(); iter; ++iter) {
					auto val = leaf.getValue(iter.offset());
					val[i] = iter.getValue();
					if (val[i] != 0) {
						vcounter++;
					}
					leaf.setValueOn(iter.offset(), val);
				}
			}
		}
	};

	openvdb::tree::LeafManager<openvdb::Vec3fTree> leafman{ out_v->tree() };
	leafman.foreach(filler);
}

void packed_FloatGrid3::swap(packed_FloatGrid3& other)
{
	v[0].swap(other.v[0]);
	v[1].swap(other.v[1]);
	v[2].swap(other.v[2]);
	m_transform.swap(other.m_transform);
}

packed_FloatGrid3 packed_FloatGrid3::deepCopy() const {
	packed_FloatGrid3 result;
	result.v[0] = v[0]->deepCopy();
	result.v[1] = v[1]->deepCopy();
	result.v[2] = v[2]->deepCopy();
	return result;
}

packed_FloatGrid3 packed_FloatGrid3::fullCopy() const {
	packed_FloatGrid3 result = deepCopy();
	result.m_gridclass = m_gridclass;
	result.m_transform = m_transform;
	return result;
}
