
#include "roads/kdtree.h"
#include <stack>
#include <utility>

using namespace roads;
using namespace Eigen;

KDTreeNode::KDTreeNode(VectorXf Value) : Point(std::move(Value)) {}

KDTreeNode* KDTree::BuildKdTree_Impl(ArrayList<VectorXf> Data, int64_t Lower, int64_t Upper, uint32_t Depth) {
    if (Lower >= Upper) {
        return nullptr;
    }

    int64_t Axis = Depth % Data[0].size();
    auto MiddleIter = Data.begin() + Lower + (Upper - Lower) / 2;
    std::nth_element(
        Data.begin() + Lower, MiddleIter, Data.begin() + Upper,
        [Axis](const Eigen::VectorXf &a, const Eigen::VectorXf &b) {
            return a[Axis] < b[Axis];
        });
    MiddleIter = Data.begin() + Lower + (Upper - Lower) / 2;

    auto* Node = new KDTreeNode(*MiddleIter);
    Node->Left = std::shared_ptr<KDTreeNode>(BuildKdTree_Impl(Data, Lower, MiddleIter - Data.begin(), Depth + 1));
    Node->Right = std::shared_ptr<KDTreeNode>(BuildKdTree_Impl(Data, MiddleIter - Data.begin() + 1, Upper, Depth + 1));

    return Node;
}

KDTree* KDTree::BuildKdTree(const ArrayList<VectorXf>& Data) {
    auto* NewTree = new KDTree;
    auto* Root = (BuildKdTree_Impl(Data, 0, int64_t(Data.size() - 1), 0));
    NewTree->Root = std::shared_ptr<KDTreeNode>(Root);

    if (!NewTree->Root) {
        throw std::invalid_argument("[Roads] KdTree built with invalid arguments.");
    }

    return NewTree;
}

void KDTree::Insert(VectorXf Point) {
    Root = Insert_Impl(Root, std::move(Point), 0);
}

std::shared_ptr<KDTreeNode> KDTree::Insert_Impl(const std::shared_ptr<KDTreeNode> &Node, VectorXf Point, uint32_t Depth) {
    if (!Node) {
        return std::make_shared<KDTreeNode>(Point);
    }

    uint32_t Axis = Depth % Point.rows();

    if (Point[Axis] < Node->Point[Axis]) {
        Node->Left = Insert_Impl(Node->Left, Point, Depth + 1);
    } else {
        Node->Right = Insert_Impl(Node->Right, Point, Depth + 1);
    }

    return Node;
}

void KDTree::SearchNode(KDTreeNode *Node, VectorXf Point, float Radius, ArrayList<VectorXf> &OutPoints, uint32_t Depth) {
    if (nullptr == Node) return;

    uint32_t Axis = Depth % Point.rows();

    float Distance = (Node->Point - Point).norm();

    if (Distance <= Radius) {
        OutPoints.push_back(Node->Point);
    }

    if (Node->Point[Axis] >= Point[Axis] - Radius) {
        SearchNode(Node->Left.get(), Point, Radius, OutPoints, Depth + 1);
    }
    if (Node->Point[Axis] <= Point[Axis] + Radius) {
        SearchNode(Node->Right.get(), Point, Radius, OutPoints, Depth + 1);
    }
}

ArrayList<VectorXf> KDTree::SearchRadius(const VectorXf &Point, float Radius) {
    ArrayList<VectorXf> Result;

    SearchNode(Root.get(), Point, Radius, Result);

    return Result;
}

Octree::Octree(float minX, float maxX, float minY, float maxY, float minZ, float maxZ, unsigned int maxDepth) :
    maxDepth(maxDepth) {
    std::array<float, 6> bounds = {minX, maxX, minY, maxY, minZ, maxZ};
    root = std::make_unique<Node>(bounds);
}

void Octree::addPoint(Octree::Point3D *point) {
    addPoint(root.get(), point, 1);
}

std::vector<Octree::Point3D *> Octree::findNearestNeighbours(float x, float y, float z, unsigned int k) {
    Point3D point(x, y, z);
    std::vector<Point3D*> nearestNeighbours(k);
    findNearestNeighbours(root.get(), &point, nearestNeighbours, 1);
    return nearestNeighbours;
}

void Octree::findNearestNeighbours(Octree::Node *node, Octree::Point3D *target, std::vector<Point3D *> &nearestNeighbours, unsigned int depth) {
    if(node == nullptr){
        return;
    }

    if(node->point){
        Point3D& current = *node->point;
        float dist = (current - *target).squaredNorm();

        for(auto& neighbour : nearestNeighbours){
            // 首先，确保即将添加的点不是已经添加到 `nearestNeighbours` 中的点。
            if(neighbour == node->point){
                return;
            }

            if(neighbour == nullptr || dist < (*neighbour - *target).squaredNorm()){
                neighbour = node->point;
                dist = (*neighbour - *target).squaredNorm();
            }
        }
    }

    if(depth < maxDepth){
        for(auto& child : node->children){
            findNearestNeighbours(child.get(), target, nearestNeighbours, depth + 1);
        }
    }
}

void Octree::addPoint(Octree::Node *node, Octree::Point3D *point, unsigned int depth) {
    if(depth >= maxDepth){
        node->point = point;
    } else {
        unsigned index = 0;
        index |= (*point)[0] > node->bounds[0] ? 1 : 0;
        index |= (*point)[1] > node->bounds[2] ? 2 : 0;
        index |= (*point)[2] > node->bounds[4] ? 4 : 0;

        if(node->children[index] == nullptr){
            std::array<float, 6> bounds{};
            bounds[0] = (index & 1 ? node->bounds[0] : (*point)[0]);
            bounds[1] = (index & 1 ? (*point)[0] : node->bounds[1]);
            bounds[2] = (index & 2 ? node->bounds[2] : (*point)[1]);
            bounds[3] = (index & 2 ? (*point)[1] : node->bounds[3]);
            bounds[4] = (index & 4 ? node->bounds[4] : (*point)[2]);
            bounds[5] = (index & 4 ? (*point)[2] : node->bounds[5]);

            node->children[index] = std::make_unique<Node>(bounds);
        }

        addPoint(node->children[index].get(), point, depth + 1);
    }
}

std::vector<Octree::Point3D *> Octree::findPointsInRadius(float x, float y, float z, float radius) {
    Point3D center(x, y, z);
    std::vector<Point3D*> pointsInRadius;
    findPointsInRadius(root.get(), &center, radius, pointsInRadius);
    return pointsInRadius;
}

void Octree::findPointsInRadius(Octree::Node *node, Octree::Point3D *center, float radius, std::vector<Point3D *> &pointsInRadius) {
    if(node == nullptr){
        return;
    }

    if(node->point){
        Point3D& current = *node->point;
        float dist = (*center - current).squaredNorm();

        if(dist <= radius * radius){
            pointsInRadius.push_back(node->point);
        }
    }

    float deltaX = std::max(node->bounds[0] - center->x(), center->x() - node->bounds[1]);
    float deltaY = std::max(node->bounds[2] - center->y(), center->y() - node->bounds[3]);
    float deltaZ = std::max(node->bounds[4] - center->z(), center->z() - node->bounds[5]);

    if (deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ < radius * radius) {
        for(auto& child : node->children){
            findPointsInRadius(child.get(), center, radius, pointsInRadius);
        }
    }
}
