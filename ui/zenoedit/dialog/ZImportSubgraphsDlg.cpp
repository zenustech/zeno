#include "ZImportSubgraphsDlg.h"
#include <zenoui/style/zenostyle.h>

CheckBoxHeaderView::CheckBoxHeaderView(Qt::Orientation orientation, QWidget* parent) : QHeaderView(orientation, parent)
{
}

void CheckBoxHeaderView::setCheckState(QVector<int> columns, bool state)
{
    for (const auto &col : columns)
    {
        m_checkedMap[col] = state;
    }
    update();
}

void CheckBoxHeaderView::paintSection(QPainter* painter, const QRect& rect, int logicalIndex) const
{
    if (m_checkedMap.contains(logicalIndex))
    {
        QStyleOptionButton option;
        int size = ZenoStyle::dpiScaled(20);
        option.rect = QRect(rect.x(), rect.y(), size, size);
        if (m_checkedMap[logicalIndex])
        {
            option.state = QStyle::State_On;
        }
        else
        {
            option.state = QStyle::State_Off;
        }
        QCheckBox checkbox;
        this->style()->drawControl(QStyle::CE_CheckBox, &option, painter, &checkbox);
        int diff = size + ZenoStyle::dpiScaled(4);
        QPen pen;
        pen.setColor(QColor(166, 166, 166));
        painter->setPen(pen);
        painter->drawText(rect.adjusted(diff, 0, diff, 0), model()->headerData(logicalIndex, Qt::Horizontal).toString());
        
        pen.setColor(QColor(115, 123, 133));
        pen.setWidthF(ZenoStyle::dpiScaled(1));
        painter->setPen(pen);
        painter->drawLine(rect.bottomLeft(), rect.bottomRight());
    }
    else
    {
        QHeaderView::paintSection(painter, rect, logicalIndex);
    }
}

void CheckBoxHeaderView::mousePressEvent(QMouseEvent* event)
{
    int index = visualIndexAt(event->pos().x());
    if (m_checkedMap.contains(index))
    {
        m_checkedMap[index] = !m_checkedMap[index];
        this->updateSection(index);
        emit signalCheckStateChanged(index, m_checkedMap[index]);
    }
    emit QHeaderView::sectionClicked(visualIndexAt(event->pos().x()));
    QHeaderView::mousePressEvent(event);
}

//SubgraphsListDlg
ZSubgraphsListDlg::ZSubgraphsListDlg(const QStringList& lst, QWidget* parent)
    : QDialog(parent)
    , m_pTableWidget(nullptr)
{
    initUI(lst);
}

ZSubgraphsListDlg::~ZSubgraphsListDlg()
{
}

void ZSubgraphsListDlg::initUI(const QStringList& subgraphs)
{
    setWindowFlags(Qt::Dialog | Qt::WindowCloseButtonHint);
    setWindowTitle(tr("List of subgraphs"));
    setAttribute(Qt::WA_DeleteOnClose, true);
    setMinimumHeight(ZenoStyle::dpiScaled(500));
    QVBoxLayout* layout = new QVBoxLayout(this);

    //inti TableWIdget
    m_pTableWidget = new QTableWidget(this);
    m_pTableWidget->setProperty("cssClass", "select_subgraph");
    CheckBoxHeaderView* pHeaderView = new CheckBoxHeaderView(Qt::Horizontal, m_pTableWidget);
    m_pTableWidget->setHorizontalHeader(pHeaderView);
    m_pTableWidget->verticalHeader()->setVisible(false);
    m_pTableWidget->setColumnCount(3);
    QStringList labels = { tr("Name"), tr("Replace"), tr("Rename") };
    m_pTableWidget->setHorizontalHeaderLabels(labels);
    pHeaderView->setCheckState(QVector<int>() << 1 << 2, false);
    connect(pHeaderView, &CheckBoxHeaderView::signalCheckStateChanged, this, [=](int col, bool bChecked) {
        updateCheckState(col, bChecked);
    if (bChecked)
        pHeaderView->setCheckState(QVector<int>() << (col == 1 ? 2 : 1), false);
    });
    m_pTableWidget->setShowGrid(false);
    m_pTableWidget->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Fixed);
    m_pTableWidget->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Fixed);
    m_pTableWidget->horizontalHeader()->setDefaultSectionSize(ZenoStyle::dpiScaled(80));
    m_pTableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    for (const auto& subgraph : subgraphs)
    {
        int row = m_pTableWidget->rowCount();
        m_pTableWidget->insertRow(row);
        QTableWidgetItem* pItem = new QTableWidgetItem(subgraph);
        m_pTableWidget->setItem(row, 0, pItem);
        QCheckBox* pReplaceCheckBox = new QCheckBox(m_pTableWidget);
        m_pTableWidget->setCellWidget(row, 1, pReplaceCheckBox);
        QCheckBox* pRenameCheckBox = new QCheckBox(m_pTableWidget);
        m_pTableWidget->setCellWidget(row, 2, pRenameCheckBox);
        connect(pReplaceCheckBox, &QCheckBox::stateChanged, this, [=]() {
            if (pReplaceCheckBox->isChecked() && pRenameCheckBox->isChecked())
            pRenameCheckBox->setChecked(false);
        });
        connect(pRenameCheckBox, &QCheckBox::stateChanged, this, [=]() {
            if (pRenameCheckBox->isChecked() && pReplaceCheckBox->isChecked())
            pReplaceCheckBox->setChecked(false);
        });
    }
    //update table width
    int width = 0;
    for (int col = 0; col < m_pTableWidget->columnCount(); col++)
    {
        width += m_pTableWidget->columnWidth(col);
    }
    m_pTableWidget->setMinimumWidth(width + ZenoStyle::dpiScaled(15));
    layout->addWidget(m_pTableWidget);

    //init Ok & Cancel button
    QHBoxLayout* pBtnLayout = new QHBoxLayout;
    QPushButton* pOk = new QPushButton(tr("OK"), this);
    QPushButton* pCancel = new QPushButton(tr("Cancel"), this);
    QSize size(ZenoStyle::dpiScaledSize(QSize(100, 30)));
    pOk->setFixedSize(size);
    pCancel->setFixedSize(size);
    pBtnLayout->addWidget(pOk);
    pBtnLayout->addWidget(pCancel);
    layout->addLayout(pBtnLayout);
    connect(pOk, &QPushButton::clicked, this, &ZSubgraphsListDlg::onOkBtnClicked);
    connect(pCancel, &QPushButton::clicked, this, &QDialog::reject);
}

void ZSubgraphsListDlg::onOkBtnClicked()
{
    QStringList replaceLst;
    QStringList renameLst;
    for (int row = 0; row < m_pTableWidget->rowCount(); row++)
    {
        if (QWidget* pReplaceWidget = m_pTableWidget->cellWidget(row, 1))
        {
            if (QCheckBox* pCheckBox = dynamic_cast<QCheckBox*>(pReplaceWidget))
            {
                pCheckBox->isChecked();
            }
        }
        if (QWidget* pRenameWidget = m_pTableWidget->cellWidget(row, 2))
        {
            if (QCheckBox* pCheckBox = dynamic_cast<QCheckBox*>(pRenameWidget))
            {
                QString name = m_pTableWidget->item(row, 0)->data(Qt::DisplayRole).toString();
                if (pCheckBox->isChecked())
                {
                    renameLst << name;
                }
                else if (QWidget* pReplaceWidget = m_pTableWidget->cellWidget(row, 1))
                {
                    if (QCheckBox* pReplaceCheckBox = dynamic_cast<QCheckBox*>(pReplaceWidget))
                    {
                        if (pReplaceCheckBox->isChecked())
                        {
                            replaceLst << name;
                        }
                    }
                }
            }
        }
    }
    emit this->selectedSignal(renameLst, true);
    emit this->selectedSignal(replaceLst, false);
    this->accept();
}

void ZSubgraphsListDlg::updateCheckState(int col, bool state)
{
    for (int row = 0; row < m_pTableWidget->rowCount(); row++)
    {
        if (QCheckBox* pCheckBox = dynamic_cast<QCheckBox*>(m_pTableWidget->cellWidget(row, col)))
        {
            pCheckBox->setChecked(state);
        }
    }
}


ZImportSubgraphsDlg::ZImportSubgraphsDlg(const QStringList& lst, QWidget *parent)
    : QDialog(parent)
    , m_subgraphs(lst)
{
    m_ui = new Ui::ZImportSubgraphsDlg;
    m_ui->setupUi(this);
    this->setWindowFlags(Qt::Dialog | Qt::WindowCloseButtonHint);
    m_ui->m_replaceBtn->setProperty("cssClass", "select_subgraph");
    m_ui->m_skipBtn->setProperty("cssClass", "select_subgraph");
    m_ui->m_selectBtn->setProperty("cssClass", "select_subgraph");
    m_ui->m_tipLabel->setText(tr("include %1 subgraphs with the same name").arg(lst.size()));
    connect(m_ui->m_replaceBtn, &QPushButton::clicked, this, [=]() {
        emit selectedSignal(lst, false);
        accept();
    });
    connect(m_ui->m_skipBtn, &QPushButton::clicked, this, [=]() {
        emit selectedSignal(QStringList(), false);
        accept();
    });
    connect(m_ui->m_selectBtn, &QPushButton::clicked, this, &ZImportSubgraphsDlg::onSelectBtnClicked);
}

ZImportSubgraphsDlg::~ZImportSubgraphsDlg()
{
}

void ZImportSubgraphsDlg::onSelectBtnClicked()
{
    ZSubgraphsListDlg*dlg = new ZSubgraphsListDlg(m_subgraphs, this);
    connect(dlg, &ZSubgraphsListDlg::selectedSignal, this, &ZImportSubgraphsDlg::selectedSignal);
    if (dlg->exec() == QDialog::Accepted)
    {
        this->accept();
    }
}