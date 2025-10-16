import QtQuick                   2.12
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3


Item {
    id: root
    property var model
    property var index
    implicitWidth: mainlayout.implicitWidth
    implicitHeight: mainlayout.implicitHeight

    RowLayout {
        id: mainlayout
        anchors.fill: parent

        Text {
            text: root.model.data(index)
            color: "black"
        }
    }
}