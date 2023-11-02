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
// \file	ColorPopup.qml
// \author	benoit@destrat.io
// \date	2017 12 13
//-----------------------------------------------------------------------------

import QtQuick                   2.7
import QtQuick.Controls          2.0
import QtQuick.Layouts           1.3

Popup {
    id: colorPopup
    width: 570
    height: 265
    property var selectedColor: undefined
    GridView {
        id: colorGridView
        anchors.fill: parent
        cellWidth: 60; cellHeight: 40
        clip: true
        property var colorModel: ListModel{ dynamicRoles: true }
        model:  colorModel
        Component.onCompleted: {
            // Create color model using HSV
            var H = [0., 300., 240., 180., 120., 60.] //  red/violet/darkblue/lightblue/green/yellow
            var S = [100 / 255., 175. / 255., 1.0]
            var V = [100 / 255., 175. / 255., 1.0]

            for ( var h = 0; h < H.length; h++ ) {
                for ( var s = S.length - 1; s >= 0; s-- )
                    for ( var v = 0; v < V.length; v++ )
                        colorModel.append( { "cellColor": Qt.hsva(H[h] / 360., S[s], V[v], 1.0) } )
            }
        }
        delegate: Component {
            id: colorCheckBox
            Item {
                width: colorGridView.cellWidth; height: colorGridView.cellHeight
                RowLayout {
                    anchors.fill: parent
                    Item { Layout.fillWidth: true } // Space eater
                    Rectangle {
                        id: colorPreview
                        Layout.preferredWidth: 24; Layout.preferredHeight: 24
                        radius: 3; color: cellColor
                        border.width: 1; border.color: Qt.darker(cellColor)
                        MouseArea { anchors.fill: parent; onClicked: colorCb.onClicked() }
                    }
                    CheckBox {
                        id: colorCb
                        Layout.preferredWidth: 24; Layout.preferredHeight: 24
                        checked: colorPopup.selectedColor === cellColor
                        onClicked: {
                            colorPopup.selectedColor = cellColor
                            colorPopup.close();
                        }
                    }
                }
            }
        } // Component: delegate
    }
}
