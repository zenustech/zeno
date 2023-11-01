import QtQuick                   2.3
import QtQuick.Controls          2.1

Rectangle {
    id:slid
    width: 200
    height: 30
    property alias backGroundColor:slid.color

    Slider {
        id:sld1
        anchors.fill:parent
    }
}
