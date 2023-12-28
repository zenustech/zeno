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

ZMaterialInfoSettingDlg::ZMaterialInfoSettingDlg(const QString& json, QWidget* parent)
    : ZFramelessDialog(parent)
    , m_jsonStr(json)
{
    ui.setupUi(this);
    QString path = ":/icons/zeno-logo.png";
    this->setTitleIcon(QIcon(path));
    this->setTitleText(tr("Match Settings"));
    this->setMainWidget(ui.m_mainWidget);
    resize(ZenoStyle::dpiScaledSize(QSize(500, 400)));
    m_doc.Parse(json.toUtf8());
    initNames();
    initKeys();
    initMatch();
    initMaterialPath();
    initButtons();
    ui.m_mainWidget->layout()->setAlignment(Qt::AlignTop);
    ui.m_mainWidget->layout()->setSpacing(ZenoStyle::dpiScaled(10));
}

ZMaterialInfoSettingDlg::~ZMaterialInfoSettingDlg()
{}

bool ZMaterialInfoSettingDlg::eventFilter(QObject* watch, QEvent* event)
{
    if (watch == ui.m_keyTableWidget->viewport() && event->type() == QEvent::MouseButtonRelease)
    {
        QWidget* pWidget = ui.m_keyTableWidget->indexWidget(ui.m_keyTableWidget->currentIndex());
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
    QStringList nameLst = ui.m_namesEdit->text().split(",");
    for (int row = 0; row < ui.m_keyTableWidget->rowCount(); row++)
    {
        QString keys = ui.m_keyTableWidget->item(row, 1)->text();
        if (!keys.isEmpty())
        {
            QRegularExpression rx(keys);
            rx.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
            for (const QString& name : nameLst)
            {
                if (rx.match(name).hasMatch())
                {
                    QString preSetName = ui.m_keyTableWidget->item(row, 0)->text();
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
    ui.m_namesEdit->setPlaceholderText(tr("Separated by ',', such as: N1, N2, N3..."));
    if (m_doc.IsObject() && m_doc.HasMember("materials"))
    {
        auto arry = m_doc["materials"].GetArray();
        QString str;
        for (int i = 0; i < arry.Size(); i++)
        {
            const QString& name = QString::fromStdString(arry[i].GetString());
            str += name;
            if (i != arry.Size() - 1)
                str += ",";
        }
        ui.m_namesEdit->setText(str);
    }
}
void ZMaterialInfoSettingDlg::initKeys()
{
    ui.m_keyTableWidget->verticalHeader()->setVisible(false);
    //ui.m_keyTableWidget->setProperty("cssClass", "select_subgraph");
    ui.m_keyTableWidget->setColumnCount(2);
    QStringList labels = { tr("Preset Subgraph"), tr("key words") };
    ui.m_keyTableWidget->setHorizontalHeaderLabels(labels);
    ui.m_keyTableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    ui.m_keyTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    for (const auto& subgIdx : pGraphsModel->subgraphsIndice(SUBGRAPH_PRESET))
    {
        int row = ui.m_keyTableWidget->rowCount();
        ui.m_keyTableWidget->insertRow(row);
        QString name = subgIdx.data(ROLE_OBJNAME).toString();
        QTableWidgetItem* pItem = new QTableWidgetItem(name);
        pItem->setFlags(pItem->flags() & ~Qt::ItemIsEditable);
        ui.m_keyTableWidget->setItem(row, 0, pItem);

        QTableWidgetItem* pKeyItem = new QTableWidgetItem();
        ui.m_keyTableWidget->setItem(row, 1, pKeyItem);
        if (m_doc.IsObject() && m_doc.HasMember("keys"))
        {
            const auto& value = m_doc["keys"];
            if (value.HasMember(name.toUtf8()))
            {
                pKeyItem->setText(QString::fromStdString(value[name.toStdString().c_str()].GetString()));
            }
        }
    }
    if (ui.m_keyTableWidget->rowCount() > 0)
    {
        int height = ui.m_keyTableWidget->rowHeight(0) * ui.m_keyTableWidget->rowCount();
        int hearderH = ui.m_keyTableWidget->horizontalHeader()->height();
        ui.m_keyTableWidget->setMinimumHeight(height + hearderH);
    }
    ui.m_keyTableWidget->viewport()->installEventFilter(this);
    // ui.m_keyTableWidget->setMouseTracking(true);
}

void ZMaterialInfoSettingDlg::initMaterialPath()
{
    CALLBACK_SWITCH cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn); //deal with ubuntu dialog slow problem when update viewport.
    };
    m_materialPath = new ZPathEdit(cbSwitch, this);
    m_materialPath->setProperty("control", CONTROL_READPATH);
    ui.m_pathLayout->addWidget(m_materialPath);
    if (m_doc.IsObject() && m_doc.HasMember("materialPath"))
    {
        QString path = m_doc["materialPath"].GetString();
        if (!path.isEmpty())
        {
            m_materialPath->setText(path);
            onPathEditFinished();
        }
    }
    connect(m_materialPath, &ZLineEdit::textEditFinished, this, &ZMaterialInfoSettingDlg::onPathEditFinished);
}

void ZMaterialInfoSettingDlg::initMatch()
{
    m_pModel = new QStandardItemModel(this);
    ui.m_matchTreeView->setModel(m_pModel);
    ui.m_matchTreeView->setHeaderHidden(true);
    ui.m_matchTreeView->setMinimumHeight(ZenoStyle::dpiScaled(200));
    ui.m_matchTreeView->hide();
    ui.m_matchLabel->hide();
}

void ZMaterialInfoSettingDlg::initButtons()
{
    int width = ZenoStyle::dpiScaled(80);
    int height = ZenoStyle::dpiScaled(30);
    ui.m_okBtn->setFixedSize(width, height);
    ui.m_cancelBtn->setFixedSize(width, height);
    connect(ui.m_okBtn, &QPushButton::clicked, this, &ZMaterialInfoSettingDlg::onOKClicked);
    connect(ui.m_cancelBtn, &QPushButton::clicked, this, &ZMaterialInfoSettingDlg::reject);
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
            ui.m_matchTreeView->setIndexWidget(pChildItem->index(), pWidget);
            connect(pComboBox, &ZComboBox::currentTextChanged, this, [=](const QString& currentText) {
                pChildItem->setData(currentText, Qt::UserRole);
            });
            if (m_doc.IsObject() && m_doc.HasMember("matchInfo"))
            {
                const auto& value = m_doc["matchInfo"];
                if (value.HasMember(name.toUtf8()))
                {
                    const auto& match = value[name.toStdString().c_str()];
                    if (match.HasMember(paramName.toUtf8()))
                    {
                        QString currText = QString::fromStdString(match[paramName.toStdString().c_str()].GetString());
                        if (lst.contains(currText))
                            pComboBox->setCurrentText(currText);
                    }
                }
            }
        }
        ui.m_matchTreeView->setExpanded(pItem->index(), true);
        ui.m_matchTreeView->show();
        ui.m_matchLabel->show();
    }
}

void ZMaterialInfoSettingDlg::onOKClicked()
{
    rapidjson::StringBuffer s;
    RAPIDJSON_WRITER writer(s);
    {
        JsonObjBatch batch(writer);

        NODE_DESCS descs;
        writer.Key("materials");
        {
            JsonArrayBatch _batch(writer);
            QStringList lst = ui.m_namesEdit->text().split(",");
            for (const auto& name : lst)
            {
                writer.String(name.toUtf8());
            }
        }

        writer.Key("keys");
        {
            JsonObjBatch batchKey(writer);
            for (int row = 0; row < ui.m_keyTableWidget->rowCount(); row++)
            {
                QString keys = ui.m_keyTableWidget->item(row, 1)->text();
                if (!keys.isEmpty())
                {
                    QString jsonKey = ui.m_keyTableWidget->item(row, 0)->text();;
                    writer.Key(jsonKey.toUtf8());
                    writer.String(keys.toUtf8());
                }
            }
        }
        if (!m_materialPath->text().isEmpty())
        {
            writer.Key("materialPath");
            writer.String(m_materialPath->text().toUtf8());
        }
        if (m_pModel->rowCount() > 0)
        {
            writer.Key("matchInfo");
            {
                JsonObjBatch batchInfo(writer);
                for (int row = 0; row < m_pModel->rowCount(); row++)
                {
                    const QModelIndex& parent = m_pModel->index(row, 0);
                    int row1 = 0;
                    QString parentKey = parent.data().toString();
                    writer.Key(parentKey.toUtf8());
                    JsonObjBatch batchItem(writer);
                    {
                        while (parent.child(row1, 0).isValid())
                        {
                            QString text = parent.child(row1, 0).data(Qt::UserRole).toString();
                            QString jsonKey = parent.child(row1, 0).data().toString();
                            writer.Key(jsonKey.toUtf8());
                            writer.String(text.toUtf8());
                            row1++;
                        }
                    }
                }
            }
        }
    }
    m_jsonStr = QString::fromUtf8(s.GetString());
    accept();
}

void ZMaterialInfoSettingDlg::getMatchInfo(QString& json, QWidget* parent)
{
    ZMaterialInfoSettingDlg dlg(json, parent);
    dlg.exec();
    json = dlg.m_jsonStr;
}