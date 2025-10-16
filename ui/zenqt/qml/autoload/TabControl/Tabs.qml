// =========================================
// =============== 标签组件父类 =============
// =========================================

import QtQuick 2.15
import Qt.labs.folderlistmodel 2.15 // 读文件


Item{

    // 标签页文件加载状态的枚举
    enum Status {
        Null, // 未加载
        Loading, // 加载中
        Ready, // 已就绪
        Error // 出现异常
    }

    // ============= 布局代码 =============
    
    anchors.fill: parent

    // “加载中”提示语
    Rectangle {
        anchors.fill: parent
        z: 100
        color: "#F3F3F3"
        visible: app.tab.fileStatus != Tabs.Status.Ready

        Text {
            anchors.centerIn: parent
            text: (app.tab.fileStatus == Tabs.Status.Error) ? app.tab.errorString : "加载中……"
            font.pixelSize: 40
        }
    }
}

        // console.time("func time");
        // console.timeEnd("func time");