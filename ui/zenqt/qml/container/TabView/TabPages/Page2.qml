import QtQuick 2.15

Rectangle {
    anchors.fill: parent
    
    
    Component.onCompleted: {
        console.log("page2 实例化。")
    }
    color: "blue"
    
    Text {
        text: "this is page 2"
    }
}