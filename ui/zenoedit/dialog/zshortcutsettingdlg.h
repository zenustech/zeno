#pragma once

#include <QtWidgets>

struct ShortCutInfo;

class ZShortCutItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
        typedef QStyledItemDelegate _base;
public:
    explicit ZShortCutItemDelegate(QObject* parent = nullptr);
    // editing
    QWidget* createEditor(QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;
protected:
    bool eventFilter(QObject* object, QEvent* event) override;
};

class ZShortCutSettingDlg : public QDialog
{
    Q_OBJECT

public:
    ZShortCutSettingDlg(QWidget *parent = nullptr);
    ~ZShortCutSettingDlg();

signals:
    void layoutChangedSignal();

private slots:
    void onCurrentIndexChanged(int index);
private:
    void initUI();

private:
    QTableWidget *m_pTableWidget;
    QVector<ShortCutInfo> m_shortCutInfos;
};
