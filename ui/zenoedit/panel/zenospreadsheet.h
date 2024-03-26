//
// Created by zh on 2022/6/27.
//

#ifndef ZENO_ZENOSPREADSHEET_H
#define ZENO_ZENOSPREADSHEET_H

#include <QtWidgets>
#include "PrimAttrTableModel.h"

class ZenoSpreadsheet : public QWidget {
    Q_OBJECT

    QLabel* pStatusBar = new QLabel();
    QLabel* pPrimName = new QLabel();

public:
    ZenoSpreadsheet(QWidget* parent = nullptr);
    PrimAttrTableModel *dataModel = nullptr;
    void clear();
    void setPrim(std::string primid);
protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
private:
    void getKeyWords();
private:
    QTableView* prim_attr_view;
    QCheckBox* m_checkSortingEnabled;
    QCheckBox* m_checkStringMapping;
    QMap<QString, QString> m_keyWords;
};




#endif //ZENO_ZENOSPREADSHEET_H
