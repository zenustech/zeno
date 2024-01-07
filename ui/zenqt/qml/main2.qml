import QtQuick 2.12
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.3
import QtQuick.Controls.Styles 1.4

ApplicationWindow {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")
    TabBar {
        id: bar
        width: parent.width
        Repeater {
            model: ["First", "Second", "Third", "Fourth", "Fifth"]
            TabButton {
                text: modelData
                width: Math.max(100, bar.width / 5)
            }
        }
    }

    StackLayout {
        width: parent.width
        currentIndex: bar.currentIndex
        Item {
            id: homeTab
        }
        Item {
            id: discoverTab
        }
        Item {
            id: activityTab
        }
    }
}