import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0
import QtQuick.Controls.Styles   1.4
import QtPositioning 5.5
import QtLocation 5.6
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
        spacing: 20

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 20
            Image {
                sourceSize.width: 80
                sourceSize.height: 80
                smooth: true
                antialiasing: true
                source: "qrc:/icons/welcome_Zeno_logo.svg"
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                color: "#F0F0F0"
                font.family: "微软雅黑"
                font.pixelSize: 32
                text: "ZENO"
                fontSizeMode: Text.Fit 
                minimumPixelSize: 10
                wrapMode: Text.WordWrap
            }
        }

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            height: 1
            color: "white"
        }

        Row {
            spacing: 128

            Column {
                spacing: 35
                Column {
                    spacing: 12
                    Text {
                        color: "#F0F0F0"
                        font.pixelSize: 14
                        font.family: "微软雅黑"
                        text: "开始"
                        fontSizeMode: Text.Fit 
                        minimumPixelSize: 10
                        wrapMode: Text.WordWrap
                    }

                    Row {
                        spacing: 8
                        Image {
                            sourceSize.width: 24
                            sourceSize.height: 24
                            smooth: true
                            antialiasing: true
                            source: "qrc:/icons/file_newfile.svg"
                        }
                        Text {
                            id: text_new_project
                            color: "#0078D4"
                            font.pixelSize: 14
                            font.family: "微软雅黑"
                            text: "新建工程"
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                hoverEnabled: true
                                onClicked: {
                                    graphs.newFile();
                                }
                                onEntered: {
                                    text_new_project.font.underline = true;
                                }
                                onExited: {
                                    text_new_project.font.underline = false;
                                }
                            }
                        }
                    }
                    Row {
                        spacing: 8
                        Image {
                            sourceSize.width: 24
                            sourceSize.height: 24
                            smooth: true
                            antialiasing: true
                            source: "qrc:/icons/file_openfile.svg"
                        }
                        Text {
                            id: text_open_project
                            color: "#0078D4"
                            font.pixelSize: 14
                            font.family: "微软雅黑"
                            text: "打开工程"
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                hoverEnabled: true
                                onClicked: {
                                    graphs.openFile()
                                }
                                onEntered: {
                                    text_open_project.font.underline = true;
                                }
                                onExited: {
                                    text_open_project.font.underline = false;
                                }
                            }
                        }         
                    }
                }
                
                Column {
                    spacing: 12

                    Text {
                        color: "#F0F0F0"
                        font.pixelSize: 14
                        font.family: "微软雅黑"
                        text: "最近的文件"
                        fontSizeMode: Text.Fit 
                        minimumPixelSize: 10
                        wrapMode: Text.WordWrap
                    }

                    Column {
                        spacing: 8
                        Repeater {
                            model: graphsmanager.recentFiles()
                            delegate: Text {
                                font.pixelSize: 14
                                font.family: "微软雅黑"
                                text: modelData.split("/").pop();
                                color: "#0078D4"
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    hoverEnabled: true
                                    onClicked: {
                                        graphsmanager.openProject(modelData)
                                        // console.log("opening " + modelData)
                                    }
                                    onEntered: {
                                        parent.font.underline = true;
                                    }
                                    onExited: {
                                        parent.font.underline = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Column {
                spacing: 16

                Text {
                    color: "#F0F0F0"
                    font.pixelSize: 14
                    font.family: "微软雅黑"
                    text: "资源"
                    fontSizeMode: Text.Fit 
                    minimumPixelSize: 10
                    wrapMode: Text.WordWrap
                }

                Rectangle {
                    color: "#242824"
                    width: 280
                    height: childrenRect.y + childrenRect.height;
                    radius: 10
                    
                    Column {
                        padding: 6

                        Image {
                            sourceSize.width: 24
                            sourceSize.height: 24
                            smooth: true
                            antialiasing: true
                            source: "qrc:/icons/welcome_website.svg"
                        }
                        Text {
                            id: zenus_website
                            color: "#F0F0F0"
                            font.pixelSize: 14
                            font.family: "微软雅黑"
                            text: "泽森科工官网"
                            onLinkActivated: Qt.openUrlExternally(link)
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                hoverEnabled: true
                                onClicked: {
                                    Qt.openUrlExternally("https://zenustech.com/")
                                }
                                onEntered: {
                                    zenus_website.font.underline = true;
                                }
                                onExited: {
                                    zenus_website.font.underline = false;
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    color: "#242824"
                    width: 280
                    height: childrenRect.y + childrenRect.height
                    radius: 10
                    
                    Column {
                        padding: 6
                        Image {
                            sourceSize.width: 24
                            sourceSize.height: 24
                            smooth: true
                            antialiasing: true
                            source: "qrc:/icons/welcome_document.svg"
                        }
                        Text {
                            id: zeno_document
                            color: "#F0F0F0"
                            font.pixelSize: 14
                            font.family: "微软雅黑"
                            text: "ZENO手册"
                            onLinkActivated: Qt.openUrlExternally(link)
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                hoverEnabled: true
                                onClicked: {
                                    Qt.openUrlExternally("https://doc.zenustech.com/")
                                }
                                onEntered: {
                                    zeno_document.font.underline = true;
                                }
                                onExited: {
                                    zeno_document.font.underline = false;
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    color: "#242824"
                    width: 280
                    height: childrenRect.y + childrenRect.height
                    radius: 10
                    
                    Column {
                        padding: 6
                        Image {
                            sourceSize.width: 24
                            sourceSize.height: 24
                            smooth: true
                            antialiasing: true
                            source: "qrc:/icons/welcome_forum.svg"
                        }
                        Text {
                            id: zenus_forum
                            color: "#F0F0F0"
                            font.pixelSize: 14
                            font.family: "微软雅黑"
                            text: "论坛"
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                hoverEnabled: true
                                onClicked: {
                                    Qt.openUrlExternally("https://forums.zenustech.com/")
                                }
                                onEntered: {
                                    zenus_forum.font.underline = true;
                                }
                                onExited: {
                                    zenus_forum.font.underline = false;
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    color: "#242824"
                    width: 280
                    height: childrenRect.y + childrenRect.height
                    radius: 10
                    
                    Column {
                        padding: 6
                        Image {
                            sourceSize.width: 24
                            sourceSize.height: 24
                            smooth: true
                            antialiasing: true
                            source: "qrc:/icons/welcome_github.svg"
                        }
                        Text {
                            id: text_github
                            color: "#F0F0F0"
                            font.pixelSize: 14
                            font.family: "微软雅黑"
                            text: "Github"
                            onLinkActivated: Qt.openUrlExternally(link)
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                hoverEnabled: true
                                onClicked: {
                                    Qt.openUrlExternally("https://github.com/legobadman/zeno")
                                }
                                onEntered: {
                                    text_github.font.underline = true;
                                }
                                onExited: {
                                    text_github.font.underline = false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}