/*
 Copyright (c) 2008-2023, Benoit AUTHEMAN All rights reserved.

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
// This file is a part of the QuickQanava software library. Copyright 2023 Benoit AUTHEMAN.
//
// \file	DiamondNode.qml
// \author	benoit@destrat.io
// \date	2015 08 25
//-----------------------------------------------------------------------------

import QtQuick              2.8
import QtQuick.Layouts      1.3

import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava"       as Qan

Qan.NodeItem {
    id: diamondNode
    Layout.preferredWidth: 100
    Layout.preferredHeight: 100
    width: Layout.preferredWidth
    height: Layout.preferredHeight

    property var symbolPolygon: new Array(5)

    // Manually update node bounding shape (used for node selection and edge intersection)
    onWidthChanged: updateBoundingShape()
    onHeightChanged: updateBoundingShape()
    function updateBoundingShape() {             // rectangular node otherwise requestUpdateBoundingShape() will never be
        var w = width - 1                       // called (an a rectangular BS will be generated more efficietly from c++)
        var w2 = w / 2
        var w5 = w / 4
        var h = height - 1
        var h2 = h / 2
        symbolPolygon[0] = Qt.point( w2, 1 )
        symbolPolygon[1] = Qt.point( w2 + h2, h2 )
        symbolPolygon[2] = Qt.point( w2, h )
        symbolPolygon[3] = Qt.point( w2 - h2, h2 )
        symbolPolygon[4] = Qt.point( w2, 1 )
        diamondNode.setBoundingShape(symbolPolygon)
    }

    Qan.CanvasNodeTemplate {
        id: template
        anchors.fill: parent
        nodeItem : diamondNode
        symbol: Canvas {
            anchors.fill: parent
            z: 1
            id: nodeSymbol
            onPaint: {
                var ctx = nodeSymbol.getContext( "2d" )
                ctx.clearRect( 0, 0, width, height )
                ctx.lineWidth = diamondNode.borderWidth
                ctx.strokeStyle = diamondNode.borderColor
                ctx.beginPath( )
                var w = width - 1;  var w2 = w / 2
                var h = height - 1; var h2 = h / 2
                ctx.moveTo( w2, 1 )
                ctx.lineTo( w2 + h2, h2 )
                ctx.lineTo( w2, h )
                ctx.lineTo( w2 - h2, h2 )
                ctx.lineTo( w2, 1 )
                ctx.stroke( )
                var gradient = ctx.createLinearGradient(0, 0, width, height);
                gradient.addColorStop(0.1, Qt.lighter( diamondNode.style.backColor, 1.8 )  );
                gradient.addColorStop(0.9, diamondNode.style.backColor );
                ctx.fillStyle = gradient
                ctx.fill( )
            }
        }
    }
}
