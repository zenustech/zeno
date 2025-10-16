import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0
import QtQuick.Controls.Styles   1.4
import Qt.labs.settings 1.1

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import Zeno 1.0 as Zen


Item {
    id: welcomeui
    property variant graphs

    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0.1,0.1,0.1,1)
    }

    Column {
        anchors.centerIn: parent

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            Rectangle {
                color: "#362125";
                width: 50;
                height: 50;
            }
            Rectangle {
                color: "#531930";
                width: 50;
                height: 50;
            }
        }

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            Rectangle {
                color: "#5F32A7";
                width: 50;
                height: 50;
            }
            Rectangle {
                color: "#890012";
                width: 50;
                height: 50;
            }
            Rectangle {
                color: "#4F670B";
                width: 50;
                height: 50;
            }
            Rectangle {
                color: "#23F568";
                width: 50;
                height: 50;
            }                       
        }
        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            color: "red";
            width: 50;
            height: 50;
        }

        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            color: "blue";
            width: 50;
            height: 50;
        }

        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            color: "yellow";
            width: 50;
            height: 50;
        }

        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            color: "black";
            width: 50;
            height: 50;
        }

        /*
        Text{
            
            font.pixelSize: 24
            text: "Some Some"
            fontSizeMode: Text.Fit 
            minimumPixelSize: 10
            wrapMode: Text.WordWrap
        }
        */

    }



}