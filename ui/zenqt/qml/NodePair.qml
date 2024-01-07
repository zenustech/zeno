import QtQuick 2.3


Item {
	Edge {
        id: outputConnectEdge
        visible: true
        point1x: rect1.x
        point1y: rect1.y
        point2x: rect2.x
        point2y: rect2.y
        color: "blue"
        thickness: 2
    }

    Rectangle {
        id: rect1
        width: 50
        height: 50
        z: mouseArea1.drag.active ||  mouseArea1.pressed ? 2 : 1
        color: Qt.rgba(Math.random(), Math.random(), Math.random(), 1)
        x: Math.random() * (win.width / 2 - 100)
        y: Math.random() * (win.height - 100)
        property point beginDrag
        property bool caught: false
        border { width:2; color: "white" }
        radius: 5
        Drag.active: mouseArea1.drag.active
 
        Text {
            anchors.centerIn: parent
            text: "INDEX"
            color: "white"
        }
        MouseArea {
            id: mouseArea1
            anchors.fill: parent
            drag.target: parent
            onPressed: {
                rect1.beginDrag = Qt.point(rect1.x, rect1.y);
            }
            onReleased: {

            }
 
        }

    }

    Rectangle {
        id: rect2
        width: 50
        height: 50
        z: mouseArea2.drag.active ||  mouseArea2.pressed ? 2 : 1
        color: Qt.rgba(Math.random(), Math.random(), Math.random(), 1)
        x: Math.random() * (win.width / 2 - 100)
        y: Math.random() * (win.height - 100)
        property point beginDrag
        property bool caught: false
        border { width:2; color: "white" }
        radius: 5
        Drag.active: mouseArea2.drag.active
 
        Text {
            anchors.centerIn: parent
            text: "index"
            color: "white"
        }
        MouseArea {
            id: mouseArea2
            anchors.fill: parent
            drag.target: parent
            onPressed: {
                rect2.beginDrag = Qt.point(rect2.x, rect2.y);
            }
            onReleased: {

            }
 
        }

    }
}