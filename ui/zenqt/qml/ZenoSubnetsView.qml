import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0
import QtQuick.Controls.Styles   1.4


StackLayout {
    id: stack_subnet_views
    property variant root_graphmodel

    ZenoGraphView {
        graphModel: root_graphmodel
    }

    function arraysEqual(a, b) {
        if (a === b) return true;
        if (a == null || b == null) return false;
        if (a.length !== b.length) return false;

        // If you don't care about the order of the elements inside
        // the array, you should sort both arrays here.
        // Please note that calling sort on an array will modify that array.
        // you might want to clone your array first.

        for (var i = 0; i < a.length; ++i) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }

    function jumpTo(path_list) {

        //console.log("jumpTo start: " + path_list)

        //寻找stack_subnet_views下所有的view对应到model，和当前的路径进行比较，如果找到，直接切换，找不到就新建并加入Layout
        let path_str = ""
        for (var i = 0; i < path_list.length; i++) {
            path_str += "/"
            path_str += path_list[i]
        }
        graphsmanager.currentPath = path_str

        var n = stack_subnet_views.children.length
        //console.log("stack.length = " + n)

        for (var i = 0; i < n; i++) {
            var child = stack_subnet_views.children[i]
            if (child.graphModel == null) {
                console.log("cannot get fucking child")
                console.log("stack_subnet_views.children: " + stack_subnet_views.children)
                continue
            }
            var paths = child.graphModel.path();
            if (arraysEqual(paths, path_list)) {
                stack_subnet_views.currentIndex = i
                return;
            }
        }

        //新建子图GraphView
        const graphsview_comp = Qt.createComponent("qrc:/ZenoGraphView.qml")
        //调用treemodel通过路径获得子图graphmodel
        var subnet_model = treeModel.getGraphByPath(path_list)
        const newgraphview = graphsview_comp.createObject(stack_subnet_views, {graphModel: subnet_model})
        stack_subnet_views.currentIndex = stack_subnet_views.count - 1;
        //console.log("subnet_model is " + subnet_model)

        //跳转的信号
        newgraphview.navigateRequest.connect(function onNavigateRequest(varlst) {
            //console.log("jump to " + varlst)
            jumpTo(varlst)
        })
    }
}