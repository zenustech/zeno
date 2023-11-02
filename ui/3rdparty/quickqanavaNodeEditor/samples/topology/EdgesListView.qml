/*
 Copyright (c) 2008-2022, Benoit AUTHEMAN All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author or Destrat.io nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL AUTHOR BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

import QtQuick 2.13
import QtQuick.Controls 2.13
import QtQuick.Layouts  1.3
import QtQuick.Controls.Material 2.1

import QuickQanava      2.0 as Qan
import TopologySample   1.0 as Qan
import "qrc:/QuickQanava" as Qan

ListView {
    id: edgesList

    // PUBLIC /////////////////////////////////////////////////////////////////
    model: undefined

    // PRIVATE ////////////////////////////////////////////////////////////////
    clip: true
    spacing: 4
    focus: true
    flickableDirection: Flickable.VerticalFlick
    highlight: Rectangle {
        x: 0
        y: edgesList.currentItem != null ? edgesList.currentItem.y : 0
        width: edgesList.width
        height: edgesList.currentItem != null ? edgesList.currentItem.height : 100
        color: Material.accent
        opacity: 0.7
        radius: 3
        Behavior on y {
            SpringAnimation {
                duration: 200
                spring: 2
                damping: 0.1
            }
        }
    }
    delegate: Item {
        id: edgeDelegate
        width: ListView.view.width
        height: edgeLayout.implicitHeight
        ColumnLayout {
            anchors.fill: parent
            id: edgeLayout
            /*property string srcLabel: ""
                        property string dstLabel: ""
                        property var edgeItemData: itemData
                        onEdgeItemDataChanged: {
                            if (itemData && itemData.item) {
                                if (itemData.item.sourceItem
                                        && itemData.item.sourceItem.node)
                                    srcLabel = itemData.item.sourceItem.node.label
                                if (itemData.item.destinationItem
                                        && itemData.item.destinationItem.node)
                                    dstLabel = itemData.item.destinationItem.node.label
                                else if (itemData.item.destinationItem
                                         && itemData.item.destinationItem.node)
                                    dstLabel = itemData.item.destinationEdge.edge.label
                            } else {
                                srcLabel = ""
                                dstLabel = ""
                            }
                        }*/
            readonly property string srcLabel: itemData && itemData.item &&
                                               itemData.item.sourceItem && itemData.item.sourceItem.node ? itemData.item.sourceItem.node.label : ""
            readonly property string dstLabel: itemData && itemData.item &&
                                               itemData.item.destinationItem && itemData.item.destinationItem.node ? itemData.item.destinationItem.node.label : ""
            Label {
                text: "Label: " + itemData.label
            }
            Label {
                text: "  Src: " + parent.srcLabel
            }
            Label {
                text: "  Dst: " + parent.dstLabel
            }
        }
        MouseArea {
            anchors.fill: parent
            onClicked: {
                edgeDelegate.ListView.view.currentIndex = index
            }
        }
    } // Item: delegate
} // ListView: edgeList
