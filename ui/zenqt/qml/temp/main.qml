import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
//import MyTestImportUri  1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4


ApplicationWindow {
    id: appWindow
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    TabView {
        id: tabView
        anchors.fill: parent

        Repeater{
            id: tabs
            model: ListModel{
                //tabModel
                ListElement{ text: "Dog"; iconPath: "icon_tabTest.png" }
                ListElement{ text: "cat"; iconPath: "icon_tabTest.png" }
                ListElement{ text: "aa"; iconPath: "icon_tabTest.png" }
            }
            delegate: Tab {
                required property string text
                required property string iconPath
                title: text
                //icon: iconPath
            }
        }
                    
        style: TabViewStyle {
            tab: Item {
                    implicitWidth: Math.round(textitem.implicitWidth + image.width + 40)
                    implicitHeight: Math.round(textitem.implicitHeight + 20)
                    Rectangle {
                        anchors.fill: parent
                        anchors.bottomMargin: 2
                        radius: 0
                        border.width: 1
                        border.color: "#2B2B2B"
                        //color:"transparent"
                    }
                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: 1
                        anchors.bottomMargin: styleData.selected ? 0 : 0
                        radius: 0
                        color: styleData.selected ? "#1F1F1F" : "#181818"
                    }
                    Text {
                        id: textitem
                        anchors.fill: parent
                        anchors.leftMargin: 4 + image.width
                        anchors.rightMargin: 4
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                        color: styleData.selected ? "#FFFFFF" : "#7A9D9D"
                        text: styleData.title
                        elide: Text.ElideMiddle
                    }
                    Image {
                        id: image
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.margins: 2
                        anchors.leftMargin: 4
                        fillMode: Image.PreserveAspectFit
                        source: "qrc:/icons/icon_tabTest.png" // control.getTab(styleData.index).icon
                    }
                }//end Item
        }//end TabViewStyle
    } //end TabView
}//end ApplicationWindow