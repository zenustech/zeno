#pragma once

#include "Eigen/Dense"
#include "pch.h"
#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

namespace roads {
    using namespace Eigen;

    struct KDTreeNode : std::enable_shared_from_this<KDTreeNode> {
        VectorXf Point;
        std::shared_ptr<KDTreeNode> Left;
        std::shared_ptr<KDTreeNode> Right;

        explicit KDTreeNode(VectorXf Value);

        virtual ~KDTreeNode() = default;
    };

    class KDTree {
        std::shared_ptr<KDTreeNode> Root = nullptr;

        static KDTreeNode* BuildKdTree_Impl(ArrayList<VectorXf> Data, int64_t Lower, int64_t Upper, uint32_t Depth);

        std::shared_ptr<KDTreeNode> Insert_Impl(const std::shared_ptr<KDTreeNode> &Node, VectorXf Point, uint32_t Depth);

        void SearchNode(KDTreeNode* Node, VectorXf Point, float Radius, ArrayList<VectorXf>& OutPoints, uint32_t Depth = 0);

    public:
        void Insert(VectorXf Point);

        ArrayList<VectorXf> SearchRadius(const VectorXf& Point, float Radius);

        static KDTree* BuildKdTree(const ArrayList<VectorXf>& Data);
    };

}// namespace roads
