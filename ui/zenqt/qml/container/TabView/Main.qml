// =====================================
// =============== 主窗口 ===============
// =====================================

import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15
import Qt.labs.settings 1.1

Window {
    id: root
    visible: true
    // flags: Qt.Window | Qt.FramelessWindowHint // 无边框窗口，保留任务栏图标

    width: 1000
    height: 600
    color: "#00000000"

    // 定义全局对象，通过 app 来访问
    Item {
        id: app

        // 标签页逻辑控制器
        TabViewController { id: tab_ }
        property var tab: tab_ // 通过 app.tab 来访问

        // 持久化存储
        Settings { 
            id: settings
            fileName: "./.settings_ui.ini" // 配置文件名


            property alias openPageList: tab_.openPageList
            property alias showPageIndex: tab_.showPageIndex
            property alias barIsLock: tab_.barIsLock

            property bool refresh: false // 用于刷新
            function save(){ // 手动刷新
                refresh=!refresh
            }
        }
    }

    // 主窗口的容器，兼做边框
    Rectangle {
        id: mainContainer
        anchors.fill: parent
        color: "#999"
        radius: children[0].radius // 圆角

        // 主窗口的内容
        Rectangle {
            id: main
            anchors.centerIn: parent
            property int borderWidth: 0 // 边框宽度
            width: parent.width-borderWidth
            height: parent.height-borderWidth
            color: "#FFF"
            radius: 0 // 圆角

            // 裁切子元素，并应用圆角
            layer.enabled: true
            layer.effect: OpacityMask {
                maskSource: Rectangle {
                    width: main.width
                    height: main.height
                    radius: main.radius
                }
            }

            // 标签视图
            TabView_ { }
        }
    }

    
    
}