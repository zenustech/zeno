import QtQuick 2.15

Rectangle {
    anchors.fill: parent
    
    Component.onCompleted: {
        console.log("page1 实例化。")
    }
    color: "red"
    
    Text {
        text: "this is page 1"
    }
}