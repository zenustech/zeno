import QtQuick                   2.3
import QtQuick.Controls          2.1

Rectangle {
    id: cbRec
    width: 200
    height: 30
    property alias backGroundColor:cbRec.color

    CheckBox {
        id: checkb
        anchors.fill:parent
    }
}
