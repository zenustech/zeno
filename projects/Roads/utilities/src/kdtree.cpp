
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
