#ifndef __ZMATERIALINFO_SETTING_DLG_H__
#define __ZMATERIALINFO_SETTING_DLG_H__

#include <zenoui/comctrl/dialog/zframelessdialog.h>
namespace Ui {
    class ZMaterialInfoSettingDlgClass;
}
struct  MaterialMatchInfo
{
    QString m_names;
    QString m_keyWords;
    QString m_materialPath;
    QString m_matchInputs;
};

class ZMaterialInfoSettingDlg : public ZFramelessDialog
{
    Q_OBJECT

public:
    ZMaterialInfoSettingDlg(const MaterialMatchInfo&info, QWidget *parent = nullptr);
    ~ZMaterialInfoSettingDlg();
    static void getMatchInfo(MaterialMatchInfo& info, QWidget* parent = nullptr);
protected:
    bool eventFilter(QObject* watch, QEvent* event) override;
private slots:
    void onPathEditFinished();
    void onOKClicked();
private:
    void initNames();
    void initKeys();
    void initMaterialPath();
    void initMatch();
    void initButtons();
    void updateMatch(const QMap<QString, QSet<QString>>& map);

private:
    Ui::ZMaterialInfoSettingDlgClass* ui;
    rapidjson::Document m_doc;
    QStandardItemModel* m_pModel;
    ZPathEdit* m_materialPath;
    QString m_jsonStr;
    MaterialMatchInfo m_matchInfo;
};
#endif
