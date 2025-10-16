// MessageLabel.qml
import QtQuick 2.12

Rectangle {
    height: 50
    property string message: "debug message"
    property var msgType: ["debug", "warning", "critical"]
    color: "black"

    Column {
        anchors.fill: parent
        padding: 5.0
        spacing: 2
        Text {
            text: msgType.toString().toUpperCase() + ":"
            font.bold: msgType == "critical"
            font.family: "Terminal Regular"
            color: msgType === "warning" || msgType === "critical" ? "red" : "yellow"
            ColorAnimation on color {
                running: msgType == "critical"
                from: "red"
                to: "black"
                duration: 1000
                loops: msgType == "critical" ? Animation.Infinite : 1
            }
        }
        Text {
            text: message
            color: msgType === "warning" || msgType === "critical" ? "red" : "yellow"
            font.family: "Terminal Regular"
        }
    }

}