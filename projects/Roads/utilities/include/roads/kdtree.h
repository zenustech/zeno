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


    class Octree {
    public:
        using Point3D = Eigen::Vector3f;

    private:
        class Node {
        public:
            Point3D* point;
            std::array<float, 6> bounds; // minX, maxX, minY, maxY, minZ, maxZ
            std::array<std::unique_ptr<Node>, 8> children;

            explicit Node(std::array<float, 6>& bounds): bounds(bounds), point(nullptr), children({nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}) {}
        };

        std::unique_ptr<Node> root;
        unsigned int maxDepth;

    public:
        Octree(float minX, float maxX, float minY, float maxY, float minZ, float maxZ, unsigned int maxDepth);

        void addPoint(Point3D* point);

        std::vector<Point3D*> findNearestNeighbours(float x, float y, float z, unsigned int k);

        std::vector<Point3D*> findPointsInRadius(float x, float y, float z, float radius);

    private:
        void findNearestNeighbours(Node* node, Point3D* target, std::vector<Point3D*>& nearestNeighbours, unsigned int depth);

        void addPoint(Node* node, Point3D* point, unsigned int depth);

        void findPointsInRadius(Node* node, Point3D* center, float radius, std::vector<Point3D*>& pointsInRadius);
    };

}// namespace roads
