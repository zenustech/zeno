#pragma once
#include "openvdb/openvdb.h"

struct grid_abs_max_op {
	grid_abs_max_op(openvdb::FloatGrid::Ptr in_grid) {
		m_max = 0;
		m_grid = in_grid;
	}

	grid_abs_max_op(const grid_abs_max_op& other, tbb::split) {
		m_max = 0;
		m_grid = other.m_grid;
	}

	//used by level0 dof leafmanager
	void operator()(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
		if (!std::isfinite(m_max)) {
			return;
		}
		auto* float_leaf = m_grid->tree().probeConstLeaf(leaf.origin());
		for (auto iter = float_leaf->cbeginValueOn(); iter; ++iter) {
			if (!std::isfinite(iter.getValue())) {
				m_max = iter.getValue();
				return;
			}
			m_max = std::max(m_max, std::abs(iter.getValue()));
		}
	}

	void join(grid_abs_max_op& other) {
		if (!std::isfinite(m_max)) {
			return;
		}
		if (!std::isfinite(other.m_max)) {
			m_max = other.m_max;
			return;
		}
		m_max = std::max(m_max, other.m_max);
	}

	openvdb::FloatGrid::Ptr m_grid;
	float m_max;
};

struct grid_dot_op {
	grid_dot_op(openvdb::FloatGrid::Ptr in_a,
		openvdb::FloatGrid::Ptr in_b) {
		m_a = in_a;
		m_b = in_b;
		dp_result = 0;
	}

	grid_dot_op(const grid_dot_op& other, tbb::split) {
		m_a = other.m_a;
		m_b = other.m_b;
		dp_result = 0;
	}

	void operator()(openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
		auto* aleaf = m_a->tree().probeConstLeaf(leaf.origin());
		auto* bleaf = m_b->tree().probeConstLeaf(leaf.origin());

		for (auto iter = aleaf->cbeginValueOn(); iter; ++iter) {
			dp_result += iter.getValue() * bleaf->getValue(iter.offset());
		}
	}

	void join(grid_dot_op& other) {
		dp_result += other.dp_result;
	}

	float dp_result;
	openvdb::FloatGrid::Ptr m_a;
	openvdb::FloatGrid::Ptr m_b;
};