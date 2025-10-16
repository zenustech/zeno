import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import zeno.enum 1.0
import "./controls"


Pane {
    id: comp
    property var node: undefined
    property var nodeItem: undefined
    property bool use_tabview: false
    //implicitWidth: panel_loader.implicitWidth
    //implicitHeight: panel_loader.implicitHeight
    implicitWidth: 300
    implicitHeight: 300
    padding: 0

    property real realWidth: 0
    property real realHeight: 0

    //internal property:
    property var scrollitem: undefined
    property var output_scrollitem: undefined

    onNodeChanged: nodeItem = node ? node.item : undefined

    background: Rectangle {
        // color: "transparent" // 设置 Pane 的背景颜色
        color: Qt.rgba(51./255, 51./255, 51./255, 1.0)
    }

    function calc_content_height() {
        // console.log("scrollitem.height: " + scrollitem.height)
        // if (output_scrollitem) {
        //     console.log("output_scrollitem.height: " + output_scrollitem.height)
        // }
        var H = scrollitem.height + output_scrollitem.height + 50
        if (output_scrollitem.height > 0) {
            H += 50
        }
        return H
    }

    function calc_content_width() {
        return scrollitem.width + 32;
    }

    Component {
        id: compBlank
        Item {
            visible: false
        }
    }

    Component {
        id: compPropertyPane
        Frame {
            id: base_tabview
            anchors.fill: parent
            property var treemodel: comp.node ? comp.node.params.customParamModel() : undefined
            property var tabsindex: treemodel ? treemodel.index(0, 0) : undefined
            property var outputsindex: treemodel ? treemodel.index(1, 0) : undefined
            property var customuimodel: comp.node ? comp.node.params.customUIModel() : undefined

            background: Rectangle {
                color: "transparent" // 背景颜色（可以设置为透明或其他颜色）
                border.color: "white" // 设置边框颜色
                border.width: 0 // 设置边框宽度
            }

            Component {
                id: comp_plain_proppane

                ColumnLayout {
                    id: plainlayout

                    Text {
                        text: comp.node.label;
                        color: Qt.rgba(210.0/255, 210.0/255, 210.0/255, 1.0)//"white"
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: "white"
                    }

                    ScrollView {
                        id: scroolv
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true

                        ScrollBar.horizontal.policy: ScrollBar.AsNeeded
                        ScrollBar.vertical.policy: ScrollBar.AsNeeded
                        // ScrollBar.horizontal.policy: ScrollBar.AlwaysOn      //打开是为了方便查看scroll组件的范围
                        // ScrollBar.vertical.policy: ScrollBar.AlwaysOn     

                        ZPropPanel_Group {
                            id: prop_onlygroup
                            visible: true
                            height: implicitHeight

                            property int minContentWidth: 300       //如果拖动宽度小于这个值，就设定为这个值，其余部分以滚动方式显示, 否则，拖动宽度越大，控件组宽度就跟随变大
                            implicitWidth: Math.max(scroolv.availableWidth, minContentWidth)    //控件组的宽度跟随整个面板的宽度，拖动越宽，则控件组占满多出的控件，同时设定一个最小值。
                            //TODO: 其实可以为每个内部组件设置最小大小，然后以某种方式计算出最小大小，然后设置到这里。

                            clip: true

                            is_input_prim: true
                            model: comp.node.params     //关联的是ParamsModel

                            Component.onCompleted: {
                                comp.scrollitem = this
                            }
                        }
                    }

                    Item {
                        visible: comp.node.params.numOfOutputPrims
                        height: 32
                    }

                    Text {
                        visible: comp.node.params.numOfOutputPrims
                        text: "Output Parameter";
                        color: Qt.rgba(210.0/255, 210.0/255, 210.0/255, 1.0)//"white"
                    }

                    Rectangle {
                        visible: comp.node.params.numOfOutputPrims
                        Layout.fillWidth: true
                        height: 1
                        color: "white"
                    }                    

                    //输出的prim类型的参数组，主要是为了让用户勾选显示/隐藏输出socket.
                    ZPropPanel_Group {
                        height: implicitHeight
                        clip: true
                        is_input_prim: false
                        visible: comp.node.params.numOfOutputPrims

                        Layout.fillWidth: true
                        model: comp.node.params

                        Component.onCompleted: {
                            comp.output_scrollitem = this
                        }
                    }
                }
            }

            Component {
                id: comp_tabbase_proppane
                ColumnLayout {
                    id: tablayout
                    //anchors.fill: parent

                    Text {
                        text: comp.node.label;
                        color: Qt.rgba(210.0/255, 210.0/255, 210.0/255, 1.0)//"white"
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: "white"
                    }

                    TabBar {
                        id: property_tabbar
                        property int tabcount: base_tabview.customuimodel.tabModel().rowCount()
                        // visible: tabcount > 1
                        Repeater {
                            model: base_tabview.customuimodel.tabModel()
                            delegate: TabButton {
                                property var mtabindex: base_tabview.customuimodel.tabModel().index(index, 0)
                                text: tabname
                                width: 100
                            }
                        }
                    }

                    StackLayout {
                        id: panel_stack
                        currentIndex: property_tabbar.currentIndex

                        Repeater {
                            model: base_tabview.customuimodel.tabModel()
                            delegate: ScrollView {
                                clip: true
                                ScrollBar.horizontal.policy: ScrollBar.AsNeeded
                                ScrollBar.vertical.policy: ScrollBar.AsNeeded

                                ZPropPanel_Tab {
                                    model: groups   //ParamTabModel::roleNames()

                                    Component.onCompleted: {
                                        if (index == 0) {
                                            comp.scrollitem = this
                                        }
                                    }
                                }
                            }
                        }
                    }

                    //输出的prim类型的参数组，主要是为了让用户勾选显示/隐藏输出socket.
                    ZPropPanel_Group {
                        height: implicitHeight
                        is_input_prim: false
                        Behavior on height {
                            NumberAnimation {
                                easing.type: Easing.InOutQuad
                            }
                        }
                        //Behavior on height { NumberAnimation { duration: 200 } }

                        clip: true

                        Layout.fillWidth: true
                        model: base_tabview.customuimodel.primOutputModel()

                        Component.onCompleted: {
                            comp.output_scrollitem = this
                        }
                    }
                }
            }

            Loader {
                anchors.fill: parent
                sourceComponent: {
                    if (customuimodel != null) {
                        return comp_tabbase_proppane
                    }
                    else{
                        return comp_plain_proppane
                    }
                }
            }
        }
    }

    Loader {
        id: panel_loader
        anchors.fill: parent

        sourceComponent: {
            return comp.nodeItem ? compPropertyPane : compBlank
        }
    }
}