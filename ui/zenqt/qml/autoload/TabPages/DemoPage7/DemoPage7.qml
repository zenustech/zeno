

import QtQuick 2.15
import QtQuick.Controls 2.15

import ".."

TabPage {
    property int test: index
    // 设定值
    title: `测试页 ${test}`
    index: 7

    Rectangle {
        anchors.fill: parent
        anchors.margins: 30
        clip: true
        color: "#d5edd1"
        radius: 20

        // 随机生成颜色
        function generateLightColor() {
            let i = 30 // 值越大，色越深
            var r = Math.floor(Math.random() * i + (255-i)).toString(16);
            var g = Math.floor(Math.random() * i + (255-i)).toString(16);
            var b = Math.floor(Math.random() * i + (255-i)).toString(16);
            return "#" + r + g + b;
        }
        
        Component.onCompleted: color = generateLightColor()
        
        
        Rectangle {
            x: 20
            y: 20
            z: 10
            width: 300
            height: 60
            radius: 20
            color: "#11000000"
            Text {
                text: `测试页${test} 内容`
                anchors.centerIn: parent
                font.pixelSize: 40
            }
        }

        Rectangle {
            anchors.right: parent.right
            anchors.rightMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            width: 500
            height: parent.height - 50
            radius: 20
            color: "#11000000"
            
            Rectangle {
                anchors.fill: parent
                anchors.margins: 40
                color: "#00000000"
                ScrollView {
                    anchors.fill: parent
                    height: parent.height
                    clip: true
                    Column {
                        spacing: 10
                        Repeater {
                            model: 1000
                            Button {
                                text: "Test Button " + (index + 1)
                                width: parent.parent.width
                                height: 50
                                background: Rectangle{
                                    anchors.fill: parent
                                    color: "#99FFFFFF"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}