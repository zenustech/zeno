// =========================================
// =============== 标签页父类 ===============
// =========================================

/*

创建标签页的方法：

1. 在 TabPages 目录下新建文件夹，名称自定。然后在文件夹中创建一个同名的qml文件作为组件入口。
    例：想新增一个叫 "MyPage" 的页面，就创建如下的结构：
        TabPages/MyPage/MyPage.qml

2. MyPage.qml 中，导入父目录，然后以 TabPage 作为根元素。
3. 根元素TabPage中，title设为标签页的展示标题。用qsTr以适应多国语言。
4. index设为标签页的展示排序。
5. 在TabPage中编写标签页其余内容
    
    import ".."
    TabPage {
        title: qsTr("My Page~")
        index: 10

        // 标签页内容
    }

*/

import QtQuick 2.15


Rectangle {

    // 可设定值
    property string title: "Unknown Tab" // 标签页名称
    property int index: 123 // 在导航页的排序，越小越靠前，允许但不应该重复，自定义值不得小于1
    property string intro: "" // 简介，支持MarkDown语法  

    property var onReady: ()=>{
        // 代替 Component.onCompleted ，在该页生成后被调用
        // 可重载以实现页面自己的延迟加载，比如将长耗时操作、复杂Loader的加载等放在这里面。
        console.log("onReady : ",title);
    }

    // 不可设定
    anchors.fill: parent
    color: "#00000000"
    clip: true


    // 记录是否激活
    property bool isActive: false
    z: isActive ? 0:-1
    visible: isActive

    // Component.onCompleted: {
    //     console.log("页生成：",title);
    // }
    // Component.onDestruction: {
    //     console.log("页销毁：",title);
    // }
}