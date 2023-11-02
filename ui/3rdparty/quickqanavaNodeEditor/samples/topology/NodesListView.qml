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
    id: nodesListView

    // PUBLIC /////////////////////////////////////////////////////////////////
    model: undefined

    //! Used for the "center on node" right click context menu action (could be undefined).
    property var graphView: undefined

    // PRIVATE ////////////////////////////////////////////////////////////////
    clip: true
    spacing: 4
    focus: true
    flickableDirection: Flickable.VerticalFlick

    highlightFollowsCurrentItem: false
    highlight: Rectangle {
        visible: nodesListView.currentItem !== undefined &&
                 nodesListView.currentItem !== null
        x: 0
        y: nodesListView.currentItem ? nodesListView.currentItem.y : 0
        width: nodesListView.width
        height: nodesListView.currentItem ? nodesListView.currentItem.height : 0
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

    Menu {
        id: nodeMenu
        title: qsTr('Nodes')
        property var node: undefined
        MenuItem {
            text: qsTr("Center On") ;
            enabled: nodesListView.graphView !== undefined
            onTriggered: {
                if (nodesListView.graphView &&
                    nodeMenu.node &&
                    nodeMenu.node.item )
                    graphView.centerOn(nodeMenu.node.item)
            }
        }
    } // Menu: nodeMenu

    delegate: Item {
        id: nodeDelegate
        width: ListView.view.width
        height: 30
        Label {
            id: nodeLabel
            text: "Label: " + itemData.label
        }
        MouseArea {
            anchors.fill: nodeDelegate
            acceptedButtons: Qt.AllButtons
            onClicked: {
                nodeMenu.node = itemData
                nodesListView.currentIndex = index
                if (mouse.button == Qt.RightButton)
                    nodeMenu.popup()
            }
        }
    } // Item: nodeDelegate
} // ListView: nodesListView
