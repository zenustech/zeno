import QtQuick 2.15
import QtQuick.Layouts 1.3


Rectangle {
    Layout.fillWidth: true
    Layout.fillHeight: true
    color: "#1F1F1F"

    width: 1180
    height: 900

    Graph {
        width: 1180
        height: 900
        graphModel: nodesModel
    }
}