import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.2

ApplicationWindow {
    id: rootWindow
    visible: true
    width: 640
    height: 480
    title: qsTr("Zeno Editor")
    property var appData: applicationData

    ZenoCollection {
        anchors.fill: parent
    }
}
