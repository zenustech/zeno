/*
 Copyright (c) 2008-2017, Benoit AUTHEMAN All rights reserved.

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

import QtQuick          2.7
import QtQuick.Controls 2.0
import QtQuick.Layouts  1.3

import QuickQanava 2.0 as Qan

ApplicationWindow {
    visible: true
    width: 1024
    height: 800
    title: qsTr("Resizer Sample")

    // BottomRightResizer default configuration test --------------------------
    Rectangle {
        id: item1
        x: 25; y: 15
        width: 50; height: 50
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            text: "Size={" + parent.width + ", " + parent.height + "}"
        }
        Qan.BottomRightResizer { target: parent }
    }
    Text {
        anchors.horizontalCenter: item1.horizontalCenter
        anchors.top: item1.bottom; anchors.topMargin: 15
        width: 100

        text: "Default resizer minimumTargetSize={50x50}"
        wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
    }

    // BottomRightResizer with target < minimumSize test ----------------------
    Rectangle {
        id: item2
        anchors.left : item1.right; anchors.leftMargin: 100
        anchors.top: item1.top
        width: 25; height: 25
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            width: parent.width
            text: "Size={" + parent.width + ", " + parent.height + "}"
            wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
        }
        Qan.BottomRightResizer {
            target: parent
            minimumTargetSize.width: 75
            minimumTargetSize.height: 50
        }
    }
    Text {
        anchors.horizontalCenter: item2.horizontalCenter
        anchors.top: item2.bottom; anchors.topMargin: 15
        width: 100

        text: "Target inital size={25x25} minimumTargetSize={75x50}"
        wrapMode: Text.Wrap
        horizontalAlignment: Qt.AlignCenter
    }


    // BottomRightResizer autoHideHandler=true test ---------------------------
    Rectangle {
        id: item3
        anchors.left : item2.right; anchors.leftMargin: 100
        anchors.top: item2.top
        width: 25; height: 25
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            width: parent.width
            text: "Size={" + parent.width + ", " + parent.height + "}"
            wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
        }
        Qan.BottomRightResizer {
            target: parent
            autoHideHandler: true
        }
    }
    Text {
        anchors.horizontalCenter: item3.horizontalCenter
        anchors.top: item3.bottom; anchors.topMargin: 15
        width: 100
        text: "Resizer.autoHideHandler=true"
        wrapMode: Text.Wrap
        horizontalAlignment: Qt.AlignCenter
    }

    // BottomRightResizer large handler test ----------------------------------
    Rectangle {
        id: item4
        anchors.left : item3.right; anchors.leftMargin: 100
        anchors.top: item3.top
        width: 50; height: 50
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            width: parent.width
            text: "Size={" + parent.width + ", " + parent.height + "}"
            wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
        }
        Qan.BottomRightResizer {
            target: parent
            handlerSize.width: 15; handlerSize.height: 15
        }
    }
    Text {
        anchors.horizontalCenter: item4.horizontalCenter
        anchors.top: item4.bottom; anchors.topMargin: 15
        width: 100
        text: "Resizer.handlerSize={15,15}"
        wrapMode: Text.Wrap
        horizontalAlignment: Qt.AlignCenter
    }

    // BottomRightResizer handler color test ----------------------------------
    Rectangle {
        id: item5
        anchors.left : item4.right; anchors.leftMargin: 100
        anchors.top: item4.top
        width: 50; height: 50
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            width: parent.width
            text: "Size={" + parent.width + ", " + parent.height + "}"
            wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
        }
        Qan.BottomRightResizer {
            target: parent
            handlerSize.width: 15; handlerSize.height: 15
            handlerColor: "#50FF1515"   // Transparent red
        }
    }
    Text {
        anchors.horizontalCenter: item5.horizontalCenter
        anchors.top: item5.bottom; anchors.topMargin: 15
        width: 100

        text: "Resizer.handlerColor=\"trans red\""
        wrapMode: Text.Wrap
        horizontalAlignment: Qt.AlignCenter
    }

    // BottomRightResizer custom handler test ---------------------------------
    Rectangle {
        id: item6
        anchors.left : item5.right; anchors.leftMargin: 100
        anchors.top: item5.top
        width: 50; height: 50
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            width: parent.width
            text: "Size={" + parent.width + ", " + parent.height + "}"
            wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
        }
        Qan.BottomRightResizer {
            target: parent
            handlerSize.width: 15; handlerSize.height: 15
            handler: Rectangle { color: "red"; border.width: 4; border.color:"violet"; radius: 7 }
            handlerColor: "violet"
        }
    }
    Text {
        anchors.horizontalCenter: item6.horizontalCenter
        anchors.top: item6.bottom; anchors.topMargin: 15
        width: 200
        text: 'Resizer.handlerComponent=import QtQuick 2.6; Rectangle {color: "red"; border.width: 4; border.color:"violet"; radius: 7 }'
        wrapMode: Text.Wrap
        horizontalAlignment: Qt.AlignCenter
    }

    // BottomRightResizer target not sibling test -----------------------------
    Rectangle {
        id: item7
        x: 25
        anchors.top: item1.bottom; anchors.topMargin: 100
        width: 50; height: 50
        color: "darkgrey"
        Text {
            anchors.centerIn: parent
            width: parent.width
            text: "Size={" + parent.width + ", " + parent.height + "}"
            wrapMode: Text.Wrap; horizontalAlignment: Qt.AlignCenter
        }
    }
    Qan.BottomRightResizer { target: item7 }
    Text {
        anchors.horizontalCenter: item7.horizontalCenter
        anchors.top: item7.bottom; anchors.topMargin: 15
        width: 200

        text: 'Resizer.target != parent (ie resizer not in target sibling)'
        wrapMode: Text.Wrap
        horizontalAlignment: Qt.AlignCenter
    }
}

