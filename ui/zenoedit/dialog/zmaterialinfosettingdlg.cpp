#include "zmaterialinfosettingdlg.h"
#include <zenoui/style/zenostyle.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>
#include "variantptr.h"
#include <zenomodel/include/nodeparammodel.h>
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/gv/callbackdef.h>
#include "zenomainwindow.h"
#include "ui_zmaterialinfosettingdlg.h"
#include <zenoui/comctrl/zpathedit.h>

ZMaterialInfoSettingDlg::ZMaterialInfoSettingDlg(const MaterialMatchInfo& info, QWidget* parent)
    : ZFramelessDialog(parent)
    , m_matchInfo(info)
{
    ui = new Ui::ZMaterialInfoSettingDlgClass;
    ui->setupUi(this);
    QString path = ":/icons/zeno-logo.png";
    this->setTitleIcon(QIcon(path));
    this->setTitleText(tr("Match Settings"));
    this->setMainWidget(ui->m_mainWidget);
    resize(ZenoStyle::dpiScaledSize(QSize(500, 400)));
    initNames();
    initKeys();
    initMatch();
    initMaterialPath();
    initButtons();
    ui->m_mainWidget->layout()->setAlignment(Qt::AlignTop);
    ui->m_mainWidget->layout()->setSpacing(ZenoStyle::dpiScaled(10));
}

ZMaterialInfoSettingDlg::~ZMaterialInfoSettingDlg()
{
}

bool ZMaterialInfoSettingDlg::eventFilter(QObject* watch, QEvent* event)
{
    if (watch == ui->m_keyTableWidget->viewport() && event->type() == QEvent::MouseButtonRelease)
    {
        QWidget* pWidget = ui->m_keyTableWidget->indexWidget(ui->m_keyTableWidget->currentIndex());
        if (QLineEdit* pEdit = qobject_cast<QLineEdit*>(pWidget))
        {
            if (pEdit->placeholderText().isEmpty())
                pEdit->setPlaceholderText("Separated by '|', such as : W1|W2|W3...");
        }
    }
    return ZFramelessDialog::eventFilter(watch, event);
}

void ZMaterialInfoSettingDlg::onPathEditFinished()
{
    QString path = m_materialPath->text();
    if (path.isEmpty())
    {
        if (m_pModel->rowCount() > 0)
        {
            m_pModel->removeRows(0, m_pModel->rowCount());
        }
        return;
    }
    QFile file(path);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        zeno::log_error("cannot open file: {} ({})", path.toStdString(),
            file.errorString().toStdString());
        return;
    }
    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    if (!doc.IsObject())
    {
        zeno::log_error("json file is corrupted");
        return;
    }
    auto jsonObject = doc.GetObject();
    QMap<QString, QSet<QString>> map;
    QStringList nameLst = ui->m_namesEdit->text().split(",");
    for (int row = 0; row < ui->m_keyTableWidget->rowCount(); row++)
    {
        QString keys = ui->m_keyTableWidget->item(row, 1)->text();
        if (!keys.isEmpty())
        {
            QRegularExpression rx(keys);
            rx.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
            for (const QString& name : nameLst)
            {
                if (rx.match(name).hasMatch())
                {
                    QString preSetName = ui->m_keyTableWidget->item(row, 0)->text();
                    if (doc.HasMember(name.toUtf8()))
                    {
                        const auto& objVal = doc[name.toStdString().c_str()];
                        QSet<QString> docKeys;
                        if (map.contains(preSetName))
                            docKeys = map[preSetName];
                        for (auto iter = objVal.MemberBegin(); iter != objVal.MemberEnd(); iter++) {
                            QString docKey = iter->name.GetString();
                            docKeys << docKey;
                        }
                        map[preSetName] = docKeys;
                    }
                }
            }
        }
    }
    if (!map.isEmpty())
    {
        updateMatch(map);
    }
}

void ZMaterialInfoSettingDlg::initNames()
{
    ui->m_namesEdit->setPlaceholderText(tr("Separated by ',', such as: N1, N2, N3..."));
    if (!m_matchInfo.m_names.isEmpty())
    {
        ui->m_namesEdit->setText(m_matchInfo.m_names);
    }
    connect(ui->m_namesEdit, &ZLineEdit::textEditFinished, this, &ZMaterialInfoSettingDlg::onPathEditFinished);
}
void ZMaterialInfoSettingDlg::initKeys()
{
    ui->m_keyTableWidget->verticalHeader()->setVisible(false);
    //ui->m_keyTableWidget->setProperty("cssClass", "select_subgraph");
    ui->m_keyTableWidget->setColumnCount(2);
    QStringList labels = { tr("Preset Subgraph"), tr("key words") };
    ui->m_keyTableWidget->setHorizontalHeaderLabels(labels);
    ui->m_keyTableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    ui->m_keyTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    for (const auto& subgIdx : pGraphsModel->subgraphsIndice(SUBGRAPH_PRESET))
    {
        int row = ui->m_keyTableWidget->rowCount();
        ui->m_keyTableWidget->insertRow(row);
        QString name = subgIdx.data(ROLE_OBJNAME).toString();
        QTableWidgetItem* pItem = new QTableWidgetItem(name);
        pItem->setFlags(pItem->flags() & ~Qt::ItemIsEditable);
        ui->m_keyTableWidget->setItem(row, 0, pItem);

        QTableWidgetItem* pKeyItem = new QTableWidgetItem();
        ui->m_keyTableWidget->setItem(row, 1, pKeyItem);
        if (!m_matchInfo.m_keyWords.isEmpty())
        {
            rapidjson::Document doc;
            doc.Parse(m_matchInfo.m_keyWords.toUtf8());
            if (doc.IsObject() && doc.HasMember(name.toUtf8()))
            {
                pKeyItem->setText(QString::fromStdString(doc[name.toStdString().c_str()].GetString()));
            }
        }
    }
    int height = ui->m_keyTableWidget->rowHeight(0) * 5;
    int hearderH = ui->m_keyTableWidget->horizontalHeader()->height();
    ui->m_keyTableWidget->setMinimumHeight(height + hearderH);
    ui->m_keyTableWidget->setMaximumHeight(ZenoStyle::dpiScaled(450));
    ui->m_keyTableWidget->viewport()->installEventFilter(this);
    connect(ui->m_keyTableWidget, &QTableWidget::itemChanged, this, &ZMaterialInfoSettingDlg::onPathEditFinished);
}

void ZMaterialInfoSettingDlg::initMaterialPath()
{
    CALLBACK_SWITCH cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn); //deal with ubuntu dialog slow problem when update viewport.
    };
    m_materialPath = new ZPathEdit(cbSwitch, this);
    m_materialPath->setProperty("control", CONTROL_READPATH);
    ui->m_pathLayout->addWidget(m_materialPath);
    if (!m_matchInfo.m_materialPath.isEmpty())
    {
        m_materialPath->setText(m_matchInfo.m_materialPath);
        onPathEditFinished();
    }
    connect(m_materialPath, &ZLineEdit::textEditFinished, this, &ZMaterialInfoSettingDlg::onPathEditFinished);
}

void ZMaterialInfoSettingDlg::initMatch()
{
    m_pModel = new QStandardItemModel(this);
    ui->m_matchTreeView->setModel(m_pModel);
    ui->m_matchTreeView->setHeaderHidden(true);
    ui->m_matchTreeView->setMinimumHeight(ZenoStyle::dpiScaled(200));
    ui->m_matchTreeView->setMaximumHeight(ZenoStyle::dpiScaled(450));
    ui->m_matchTreeView->hide();
    ui->m_matchLabel->hide();
}

void ZMaterialInfoSettingDlg::initButtons()
{
    int width = ZenoStyle::dpiScaled(80);
    int height = ZenoStyle::dpiScaled(30);
    ui->m_okBtn->setFixedSize(width, height);
    ui->m_cancelBtn->setFixedSize(width, height);
    connect(ui->m_okBtn, &QPushButton::clicked, this, &ZMaterialInfoSettingDlg::onOKClicked);
    connect(ui->m_cancelBtn, &QPushButton::clicked, this, &ZMaterialInfoSettingDlg::reject);
}

void ZMaterialInfoSettingDlg::updateMatch(const QMap<QString, QSet<QString>>& map)
{
    if (m_pModel->rowCount() > 0)
        m_pModel->removeRows(0, m_pModel->rowCount());
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    for (const auto& subgIdx : pGraphsModel->subgraphsIndice(SUBGRAPH_PRESET))
    {
        QString name = subgIdx.data(ROLE_OBJNAME).toString();
        if (!map.contains(name))
            continue;
        QStandardItem* pItem = new QStandardItem(name);
        m_pModel->appendRow(pItem);
        QModelIndexList nodes = pGraphsModel->searchInSubgraph("SubInput", subgIdx);
        for (const QModelIndex& subInput : nodes)
        {
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInput.data(ROLE_NODE_PARAMS));
            QModelIndex nameIdx = nodeParams->getParam(PARAM_PARAM, "name");
            QString paramName = nameIdx.data(ROLE_PARAM_VALUE).toString();
            QStandardItem* pChildItem = new QStandardItem(paramName);
            pItem->appendRow(pChildItem);
            pChildItem->setData(paramName, Qt::UserRole);
            ZComboBox* pComboBox = new ZComboBox(this);
            QList<QString> lst = map[name].toList();
            lst.prepend(paramName);
            pComboBox->addItems(lst);
            pComboBox->setCurrentText(paramName);
            QWidget* pWidget = new QWidget(this);
            QHBoxLayout* pLayout = new QHBoxLayout(pWidget);
            pLayout->setMargin(0);
            pLayout->addStretch();
            pLayout->addWidget(pComboBox);
            ui->m_matchTreeView->setIndexWidget(pChildItem->index(), pWidget);
            connect(pComboBox, &ZComboBox::currentTextChanged, this, [=](const QString& currentText) {
                pChildItem->setData(currentText, Qt::UserRole);
            });
            if (!m_matchInfo.m_matchInputs.isEmpty())
            {
                rapidjson::Document doc;
                doc.Parse(m_matchInfo.m_matchInputs.toUtf8());
                if (doc.IsObject() && doc.HasMember(name.toUtf8()))
                {
                    const auto& match = doc[name.toStdString().c_str()];
                    if (match.HasMember(paramName.toUtf8()))
                    {
                        QString currText = QString::fromStdString(match[paramName.toStdString().c_str()].GetString());
                        if (lst.contains(currText))
                            pComboBox->setCurrentText(currText);
                    }
                }
            }
        }
        ui->m_matchTreeView->setExpanded(pItem->index(), true);
        ui->m_matchTreeView->show();
        ui->m_matchLabel->show();
    }
}

void ZMaterialInfoSettingDlg::onOKClicked()
{
    if (ui->m_namesEdit->text().isEmpty())
        return;
    //names
    m_matchInfo.m_names = ui->m_namesEdit->text();

    QJsonObject keysJson;
    for (int row = 0; row < ui->m_keyTableWidget->rowCount(); row++)
    {
        QString keys = ui->m_keyTableWidget->item(row, 1)->text();
        if (!keys.isEmpty())
        {
            QString jsonKey = ui->m_keyTableWidget->item(row, 0)->text();;
            keysJson[jsonKey] = keys;
        }
    }
    QJsonDocument doc;
    if (keysJson.isEmpty())
    {
        m_matchInfo.m_keyWords = "";
    }
    else
    {
        doc.setObject(keysJson);
        m_matchInfo.m_keyWords = doc.toJson(QJsonDocument::Compact);
    }

    //material path
    m_matchInfo.m_materialPath = m_materialPath->text();

    //match inputs
    QJsonObject matchJson;
    for (int row = 0; row < m_pModel->rowCount(); row++)
    {
        const QModelIndex& parent = m_pModel->index(row, 0);
        int row1 = 0;

        QJsonObject obj;
        while (parent.child(row1, 0).isValid())
        {
            QString text = parent.child(row1, 0).data(Qt::UserRole).toString();
            QString jsonKey = parent.child(row1, 0).data().toString();
            obj[jsonKey] = text;
            row1++;
        }
        QString parentKey = parent.data().toString();
        matchJson[parentKey] = obj;
    }
    if (matchJson.isEmpty())
    {
        m_matchInfo.m_matchInputs = "";
    }
    else
    {
        doc.setObject(matchJson);
        m_matchInfo.m_matchInputs = doc.toJson(QJsonDocument::Compact);
    }
    accept();
}

void ZMaterialInfoSettingDlg::getMatchInfo(MaterialMatchInfo& info, QWidget* parent)
{
    ZMaterialInfoSettingDlg dlg(info, parent);
    dlg.exec();
    info = dlg.m_matchInfo;
}