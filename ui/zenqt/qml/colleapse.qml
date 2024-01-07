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

        ListView {
            anchors.fill: parent
            header: headerView
            footer: footerView

            model: nestedModel

            delegate:
           
        }

        ListModel {
        id: nestedModel
        ListElement {
            categoryName: "Veggies"
            collapsed: true
            subItems: [
                ListElement { itemName: "Tomato" },
                ListElement { itemName: "Cucumber" },
                ListElement { itemName: "Onion" },
                ListElement { itemName: "Brains" }
            ]
        }
    }


        Component {
            id: headerView
            Item {
                width: parent.width
                height: 30
                RowLayout {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 8
                    Text { 
                        text: "List"
                        font.bold: true
                        font.pixelSize: 20
                        Layout.preferredWidth: 120
                    }
                }            
            }
        }


        Component {
            id: footerView
            Item{
                id: footerRootItem
                width: 40
                height: 30
                signal add()
           
                Button {
                    id: addOne
                    anchors.centerIn: parent
                    text: "Add Item"
                    onClicked: footerRootItem.add()
                }
            }
        }

       

        
    }
    
