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
// This file is a part of the QuickQanava software library.
//
// \file	RectGradientBackground.qml
// \author	benoit@destrat.io
// \date	2018 03 25
//-----------------------------------------------------------------------------

import QtQuick  2.7

import QuickQanava    2.0 as Qan
import "qrc:/QuickQanava" as Qan

/*! \brief Node or group background component with gradient fill, no effect and backOpacity style support
 *
 */
Item {
    // PUBLIC /////////////////////////////////////////////////////////////////
    property var    style: undefined

    property real            backRadius:    style ? style.backRadius : 4.
    readonly property real   backOpacity:   style ? style.backOpacity : 0.8
    readonly property color  baseColor:     style ? style.baseColor: Qt.rgba(0., 0., 0., 0.)
    readonly property color  backColor:     style ? style.backColor : Qt.rgba(0., 0., 0., 0.)
    readonly property real   borderWidth:   style ? style.borderWidth : 1.
    readonly property color  borderColor:   style ? style.borderColor : Qt.rgba(1., 1., 1., 0.)

    // PRIVATE ////////////////////////////////////////////////////////////////
    // Note: Top level item is used to isolate rendering of:
    //    - background with a gradient effect
    //    - foreground (rectangle border)
    // to ensure that border is always rasterized even at high scale and never
    // batched/cached with gradient effect SG node to avoid blurry edges
    Rectangle {
        id: background
        anchors.fill: parent
        radius: backRadius
        color: Qt.rgba(0, 0, 0, 1)  // Force black, otherwise, effect does not reasterize gradient pixels
        border.width: 0             // Do not draw border, just the background gradient (border is drawn in foreground)
        antialiasing: true
        opacity: backOpacity

        layer.enabled: true
        layer.effect: Qan.LinearGradient {
            start:  Qt.point(0.,0.)
            end:    Qt.point(background.width, background.height)
            cached: false
            gradient: Gradient {
                GradientStop { position: 0.0; color: baseColor }
                GradientStop { position: 1.0;  color: backColor }
            }
        }
    }
    Rectangle {
        id: foreground
        anchors.fill: parent    // Background follow the content layout implicit size
        radius: backRadius
        color: Qt.rgba(0, 0, 0, 0)  // Fully transparent
        border.color: borderColor
        border.width: borderWidth
        antialiasing: true
        // Note: Do not enable layer to avoid aliasing at high scale
    }
}  // Item
