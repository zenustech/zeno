#ifndef __ZPREFERENCESDLG_H__
#define __ZPREFERENCESDLG_H__

#include <QtWidgets>
#include <zenoui/comctrl/dialog/zframelessdialog.h>

class ZPathEdit;
class ShortcutsPane;

//LanguagePane
class ZLanguagePane : public QWidget
{
    Q_OBJECT
public : 
    explicit ZLanguagePane(QWidget* parent = nullptr);
    void saveValue();
private:
    QRadioButton* m_pEngBtn;
    QRadioButton* m_pCNBtn;
};

//ZenoCachePane
class ZenoCachePane : public QWidget 
{
    Q_OBJECT
public:
    explicit ZenoCachePane(QWidget* parent = nullptr);
    void saveValue();
private:
    ZPathEdit *m_pPathEdit;
    QCheckBox* m_pTempCacheDir;
    QCheckBox* m_pAutoCleanCache;
    QCheckBox* m_pEnableCheckbox;
    QSpinBox* m_pCacheNumSpinBox;
    QDoubleSpinBox* m_pViewportPointSizeScaleSpinBox;

    QCheckBox* m_pEnableShiftChangeFOV;
    QSpinBox* m_pViewportSampleNumber;
};

//NASLOCPane
class ZNASLOCPane : public QWidget 
{
    Q_OBJECT
public:
    explicit ZNASLOCPane(QWidget* parent = nullptr);
    void saveValue();
private:
    ZPathEdit* m_pPthEdit;
};

//Layout Pane
class ZLayoutPane : public QWidget
{
    Q_OBJECT
public:
    explicit ZLayoutPane(QWidget* parent = nullptr);
    void saveValue();
private:
    QListWidget* m_listWidget;
};

//PreferencesTabBar
class ZPreferencesTabBar : public QTabBar 
{
    Q_OBJECT
public:
    QSize tabSizeHint(int index) const;
protected:
    void paintEvent(QPaintEvent* event);
};

class ZPreferencesTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    explicit ZPreferencesTabWidget(QWidget *parent = nullptr);

    void saveSettings();
private:
    void initUI();

private:
    ZLanguagePane* m_pLanguagePane;
    ZNASLOCPane* m_pNASLOCPane;
    ZenoCachePane* m_pZenoCachePane;
    ShortcutsPane* m_pShortcutsPane;
    ZLayoutPane* m_pLayoutPane;
};

class ZPreferencesDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    explicit ZPreferencesDlg(QWidget *parent = nullptr);
    ~ZPreferencesDlg();
private:
    void initUI();
private:
    ZPreferencesTabWidget* m_pTabWidget;
    QPushButton* m_pOKBtn;
    QPushButton* m_pCancelBtn;
};
#endif

