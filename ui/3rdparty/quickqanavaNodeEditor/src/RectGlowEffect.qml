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
// \file	RectGlowEffect.qml
// \author	benoit@destrat.io
// \date	2018 03 23
//-----------------------------------------------------------------------------

import QtQuick              2.7

import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava" as Qan

/*! \brief Node or group background glow effect with transparent mask for node content.
 */
Item {
    id: glowEffect

    // PUBLIC /////////////////////////////////////////////////////////////////
    property var    style: undefined

    // PRIVATE ////////////////////////////////////////////////////////////////
    // Default settings for rect radius, shadow margin is the _maximum_ shadow radius (+vertical or horizontal offset).
    readonly property real   glowRadius:     style ? style.effectRadius : 3.
    readonly property color  glowColor:      style ? style.effectColor : Qt.rgba(0.7, 0.7, 0.7, 0.7)
    readonly property real   effectMargin:   glowRadius * 2
    readonly property real   borderWidth:    style ? style.borderWidth : 1.
    readonly property real   borderWidth2:   borderWidth / 2.
    readonly property real   backRadius:     style ? style.backRadius : 4.

    Item {
        id: effectBackground
        z: -1   // Effect should be behind edges , docks and connectors...
        x: -effectMargin; y: -effectMargin
        width: glowEffect.width + effectMargin * 2
        height: glowEffect.height + effectMargin * 2
        visible: false
        Rectangle {
            x: effectMargin
            y: effectMargin
            width: glowEffect.width       // Avoid aliasing artifacts for mask at high scales.
            height: glowEffect.height
            radius: backRadius
            border.width: borderWidth
            color: Qt.rgba(0, 0, 0, 1)
            antialiasing: true
            layer.enabled: true
            layer.effect: Qan.Glow {
                color: glowEffect.glowColor
                radius: glowEffect.glowRadius
                samples: Math.min(16, glowEffect.glowRadius)
                spread: 0.25
                transparentBorder: true
                cached: false
                visible: glowEffect.style !== undefined ? style.effectEnabled : false
            }
        }
    }
    Item {  // Used as a mask to prevent effect beeing applied on node background, usefull if the node
        id: backgroundMask  // background is transparent
        anchors.fill: effectBackground
        visible: false
        Rectangle {
            x: effectMargin + borderWidth2
            y: effectMargin + borderWidth2
            width: glowEffect.width - borderWidth
            height: glowEffect.height - borderWidth
            radius: backRadius
            color: Qt.rgba(0,0,0,1)
            antialiasing: true
        }
    }
    Qan.OpacityMask {
        anchors.fill: effectBackground
        source: effectBackground
        maskSource: backgroundMask
        invert: true
    }
}  // Item: glowEffect
