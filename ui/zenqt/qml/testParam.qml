import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4


ApplicationWindow {
    id: appWindow
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Rectangle {
        anchors.fill: parent

        Column  {
            spacing: 0

            anchors.fill: parent
            anchors.margins: 0

            /*
            ZParam2 {
                id: param1
                name: "position"
            }*/
        
            ZParam2 {
                id: param2
                name: "scale"
            }
        }
    }
}