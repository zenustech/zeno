#include "zcefnodeseditor.h"

ZCefNodesEditor::ZCefNodesEditor(const QString url, const QCefSetting *setting, QWidget *parent)
    : QCefView(url, setting, parent)
{

}

ZCefNodesEditor::~ZCefNodesEditor() {
    int j;
    j = 0;
}