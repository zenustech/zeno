
#include "roads/kdtree.h"

using namespace roads;
using namespace Eigen;

KDTreeNode::KDTreeNode(VectorXf Value) : Point(std::move(Value)), Left(nullptr), Right(nullptr) {}

std::shared_ptr<KDTreeNode> KDTree::BuildKdTree_Impl(ArrayList<VectorXf> &Data, uint32_t Lower, uint32_t Upper, uint32_t Depth) {
    if (Lower >= Upper || Data.empty()) {
        return nullptr;
    }

    // check all data, they must have same dimension.
    const uint32_t Dim = Data[0].rows();
    if (Depth == 0 && std::any_of(std::begin(Data), std::end(Data), [Dim] (const VectorXf& Value) { return Value.rows() != Dim; })) {
        return nullptr;
    }

    uint32_t Axis = Depth % Data[0].rows();

    std::sort(std::begin(Data) + Lower, std::begin(Data) + Upper, [Axis] (const VectorXf &a, const VectorXf &b) {
        return a[Axis] < b[Axis];
    });

    uint32_t MedianIndex = (Lower + Upper) / 2;
    auto Node = std::make_shared<KDTreeNode>(Data[MedianIndex]);

    Node->Left = BuildKdTree_Impl(Data, Lower, MedianIndex, Depth + 1);
    Node->Right = BuildKdTree_Impl(Data, MedianIndex + 1, Upper, Depth + 1);

    return Node;
}

std::shared_ptr<KDTree> KDTree::BuildKdTree(ArrayList<VectorXf> Data) {
    std::shared_ptr<KDTree> NewTree = std::make_shared<KDTree>();
    NewTree->Root = NewTree->BuildKdTree_Impl(Data, 0, Data.size(), 0);

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
