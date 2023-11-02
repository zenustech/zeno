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

//-----------------------------------------------------------------------------
// This file is a part of the QuickQanava software library. Copyright 2015 Benoit AUTHEMAN.
//
// \file	FlowNode.qml
// \author	benoit@destrat.io
// \date	2017 12 12
//-----------------------------------------------------------------------------

import QtQuick              2.7
import QtQuick.Controls     2.0
import QtQuick.Layouts      1.3

import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava"   as Qan
import QuickQanava.Samples  1.0

Qan.NodeItem {
    id: operationNodeItem
    Layout.preferredWidth: 150
    Layout.preferredHeight: 70
    width: Layout.preferredWidth
    height: Layout.preferredHeight
    connectable: Qan.NodeItem.UnConnectable // Do not show visual edge connector, use out port instead

    Connections {       // Observe "node item" "node" ouput value changes andupdate out port label
        target: node
        function onOutputChanged() {
            if ( ports.itemCount > 0 )
                ports.listReference.itemAt(0).label = "OUT=" + node.output.toFixed(1)
        }
    }

    Qan.RectNodeTemplate {
        anchors.fill: parent
        nodeItem : parent
        ComboBox {
            anchors.left: parent.left; anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            anchors.margins: 4
            model: [ "+ operator", "* operator"]
            currentIndex: node.operation === OperationNode.Add ? 0 : 1
            onCurrentIndexChanged: {
                node.operation = currentIndex === 0 ? OperationNode.Add : OperationNode.Multiply
            }
        }
    }
}
