/*
 Copyright (c) 2008-2020, Benoit AUTHEMAN All rights reserved.

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

import QtQuick                   2.8
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0

import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava"   as Qan

Qan.GraphView {
    id: graphView
    anchors.fill: parent
    navigable   : true
    property var edgeItems: []
    graph: Qan.Graph {
        id: graph
        connectorEnabled: true
        Component.onCompleted: {
            var s1 = graph.insertNode();
            s1.label = "S1"; s1.item.x = 50; s1.item.y = 50
            var d1 = graph.insertNode();
            d1.label = "D1"; d1.item.x = 200; d1.item.y = 50
            var e1 = graph.insertEdge(s1, d1); //e1.style = null; e1.lineType = Qan.EdgeStyle.Ortho

            var s2 = graph.insertNode();
            s2.label = "S2"; s2.item.x = 350; s2.item.y = 50
            var d2 = graph.insertNode();
            d2.label = "D2"; d2.item.x = 350; d2.item.y = 250            
            var e2 = graph.insertEdge(s2, d2); //e2.style = null; e2.lineType = Qan.EdgeStyle.Ortho
        } // Qan.Graph.Component.onCompleted()
        onEdgeRightClicked: {
            console.error('edge=' + edge)
            console.error('pos=' + pos)
        }
    } // Qan.Graph
    CheckBox {
        anchors.top: parent.top; anchors.topMargin: 4
        anchors.right: parent.right; anchors.rightMargin: 2
        id: dashed
        text: "Ortho"
        checked: defaultEdgeStyle.lineType === Qan.EdgeStyle.Ortho
        onClicked: {
            defaultEdgeStyle.lineType = !checked ? Qan.EdgeStyle.Straight : Qan.EdgeStyle.Ortho
        }
    }
}  // Qan.GraphView
