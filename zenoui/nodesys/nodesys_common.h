#ifndef __NODESYS_COMMON_H__
#define __NODESYS_COMMON_H__

enum ZenoGVItemType {
    ZTYPE_NODE = QGraphicsItem::UserType + 1,
    ZTYPE_LINK,
    ZTYPE_FULLLINK,
    ZTYPE_TEMPLINK,
    ZTYPE_SOCKET,
    ZTYPE_IMAGE,
    ZTYPE_PARAMWIDGET
};

struct EdgeInfo {
    QString srcNode;
    QString dstNode;
    QString srcPort;
    QString dstPort;
    EdgeInfo() = default;
    EdgeInfo(const QString &srcId, const QString &dstId, const QString &srcPort, const QString &dstPort)
        : srcNode(srcId), dstNode(dstId), srcPort(srcPort), dstPort(dstPort) {}
    bool operator==(const EdgeInfo &rhs) const {
        return srcNode == rhs.srcNode && dstNode == rhs.dstNode &&
               srcPort == rhs.srcPort && dstPort == rhs.dstPort;
    }
    bool operator<(const EdgeInfo &rhs) const {
        if (srcNode != rhs.srcNode) {
            return srcNode < rhs.srcNode;
        } else if (dstNode != rhs.dstNode) {
            return dstNode < rhs.dstNode;
        } else if (srcPort != rhs.srcPort) {
            return srcPort < rhs.srcPort;
        } else if (dstPort != rhs.dstPort) {
            return dstPort < rhs.dstPort;
        } else {
            return 0;
        }
    }
};

struct cmpEdge {
    bool operator()(const EdgeInfo &lhs, const EdgeInfo &rhs) const {
        return lhs.srcNode < rhs.srcNode && lhs.dstNode < rhs.dstNode &&
               lhs.srcPort < rhs.srcPort && lhs.dstPort < rhs.dstPort;
    }
};

#endif