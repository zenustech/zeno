// =============================================================
// =============== 水平标签组件（即标签按钮位于顶部） =============
// =============================================================

import QtQuick 2.15
import QtQuick.Layouts 1.15

Tabs{
    id: tabs

    ColumnLayout{
        anchors.fill: parent
        spacing: 0

        // 标签按钮容器
        Rectangle {
            id: hTabBarContainer
            Layout.fillWidth: true
            height: 40

            color: "#F3F3F3"

            HTabBar { }
        }

        // 标签页容器，只负责展示，不负责逻辑
        Item {
            id: hTabsContainer
            Layout.fillWidth: true
            Layout.fillHeight: true

            // 获取页面内容控件
            Component.onCompleted: { 
                app.tab.pageContainer.parent = this
            }
        }
    }
}
