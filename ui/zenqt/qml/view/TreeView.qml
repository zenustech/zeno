/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021 Maurizio Ingrassia
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQml 2.15

Flickable {
    id: root

    implicitWidth: 400
    implicitHeight: 400
    clip: true

    property var model
    readonly property alias currentIndex: tree.selectedIndex
    readonly property alias currentItem: tree.currentItem
    property var currentData

    property alias handle: tree.handle
    property alias contentItem: tree.contentItem
    property Component highlight: Rectangle {
        color: root.selectedColor
    }

    property alias selectionEnabled: tree.selectionEnabled
    property alias hoverEnabled: tree.hoverEnabled

    property alias color: tree.color
    property alias handleColor: tree.handleColor
    property alias hoverColor: tree.hoverColor
    property alias selectedColor: tree.selectedColor
    property alias selectedItemColor: tree.selectedItemColor

    property alias rowHeight: tree.rowHeight
    property alias rowPadding: tree.rowPadding
    property alias rowSpacing: tree.rowSpacing

    property alias fontMetrics: tree.fontMetrics
    property alias font: tree.font

    enum Handle {
        Triangle,
        TriangleSmall,
        TriangleOutline,
        TriangleSmallOutline,
        Chevron,
        Arrow
    }

    property int handleStyle: TreeView.Handle.Triangle

    contentHeight: tree.height
    contentWidth: width
    boundsBehavior: Flickable.StopAtBounds
    ScrollBar.vertical: ScrollBar {}

    Connections { function onCurrentIndexChanged() { if(currentIndex) currentData = model.data(currentIndex) }  }

    TreeViewItem {
        id: tree

        model: root.model
        parentIndex: model.rootIndex()
        childCount: model.rowCount(parentIndex)

        itemLeftPadding: 0
        color: root.color
        handleColor: root.handleColor
        hoverColor: root.hoverColor
        selectedColor: root.selectedColor
        selectedItemColor: root.selectedItemColor
        defaultIndicator: indicatorToString(handleStyle)
        z: 1

        Connections {
           target: root.model
           ignoreUnknownSignals: true
           function onLayoutChanged() {
               tree.childCount = root.model ? root.model.rowCount(tree.parentIndex) : 0
           }
        }
    }

    Loader {
        id: highlightLoader
        sourceComponent: highlight

        width: root.width
        height: root.rowHeight
        z: 0
        visible: root.selectionEnabled && tree.currentItem !== null

        Binding {
            target: highlightLoader.item
            property: "y"
            value: tree.currentItem ? tree.currentItem.mapToItem(tree, 0, 0).y + tree.anchors.topMargin : 0
            when: highlightLoader.status === Loader.Ready
        }
    }

    function indicatorToString(handle){
        switch (handle){
            case TreeView.Handle.Triangle: return "▶";
            case TreeView.Handle.TriangleSmall: return "►";
            case TreeView.Handle.TriangleOutline: return "▷";
            case TreeView.Handle.TriangleSmallOutline: return "⊳";
            case TreeView.Handle.Chevron: return "❱";
            case TreeView.Handle.Arrow: return "➤";
            default: return "▶";
        }
    }
}
