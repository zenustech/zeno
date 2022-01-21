#include "zenosubnetlistview.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "style/zenostyle.h"
#include <model/graphsmodel.h>
#include "zsubnetlistitemdelegate.h"


ZSubnetListModel::ZSubnetListModel(GraphsModel* pModel, QObject* parent)
    : QStandardItemModel(parent)
    , m_model(pModel)
{
}

int ZSubnetListModel::rowCount(const QModelIndex& parent) const
{
    return m_model->rowCount(parent) + 1;
}

QVariant ZSubnetListModel::data(const QModelIndex& index, int role) const
{
    if (index.row() == 0)
    {
        if (role == Qt::DisplayRole)
        {
            const QString& filePath = m_model->filePath();
            QFileInfo fi(filePath);
            const QString& fn = fi.fileName();
            return fn;
        }
        else
        {
            Q_ASSERT(false);
            return QVariant();
        }
    }
    else
    {
        return m_model->data(createIndex(index.row() - 1, index.column(), index.internalId()));
    }
}

QModelIndex ZSubnetListModel::index(int row, int column, const QModelIndex& parent) const
{
    return QStandardItemModel::index(row, column, parent);
}


ZenoSubnetListView::ZenoSubnetListView(QWidget* parent)
    : QListView(parent)
{
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFrameShape(QFrame::NoFrame);
    setFrameShadow(QFrame::Plain);
}

ZenoSubnetListView::~ZenoSubnetListView()
{
}

void ZenoSubnetListView::initModel(GraphsModel* pModel)
{
    setModel(pModel);
    setItemDelegate(new ZSubnetListItemDelegate(pModel, this));
    viewport()->setAutoFillBackground(false);

    update();
}

QSize ZenoSubnetListView::sizeHint() const
{
    if (model() == nullptr)
        return QListView::sizeHint();

    if (model()->rowCount() == 0)
        return QListView::sizeHint();

    int nToShow = model()->rowCount();
    return QSize(sizeHintForColumn(0), nToShow * sizeHintForRow(0));
}

void ZenoSubnetListView::paintEvent(QPaintEvent* e)
{
    QListView::paintEvent(e);
}


///////////////////////////////////////////////////////////////////////////////////
ZenoSubnetListPanel::ZenoSubnetListPanel(QWidget* parent)
    : QWidget(parent)
    , m_pListView(nullptr)
{
    QVBoxLayout* pMainLayout = new QVBoxLayout;

    QHBoxLayout* pLabelLayout = new QHBoxLayout;
    QLabel* pIcon = new QLabel;

    m_pTextLbl = new QLabel("");
    m_pTextLbl->setFont(QFont("HarmonyOS Sans", 12));
    QPalette pal = m_pTextLbl->palette();
    pal.setColor(QPalette::WindowText, QColor(128, 124, 122));
    m_pTextLbl->setPalette(pal);

    pLabelLayout->addWidget(pIcon);
    pLabelLayout->addWidget(m_pTextLbl);
    pLabelLayout->addStretch();
    pLabelLayout->setContentsMargins(0, 0, 0, 0);
    pMainLayout->addLayout(pLabelLayout);

    m_pListView = new ZenoSubnetListView;
    pMainLayout->addWidget(m_pListView);

    QLabel* pNewSubnetBtn = new QLabel("Add New Subnet");
    pNewSubnetBtn->setFont(QFont("HarmonyOS Sans", 13));
    pal = pNewSubnetBtn->palette();
    pal.setColor(QPalette::WindowText, QColor(116, 116, 116));
    pal.setColor(pNewSubnetBtn->backgroundRole(), QColor(56, 57, 56));
    pNewSubnetBtn->setAutoFillBackground(true);
    pNewSubnetBtn->setAlignment(Qt::AlignCenter);
    pNewSubnetBtn->setPalette(pal);
    pNewSubnetBtn->setFixedHeight(ZenoStyle::dpiScaled(40));

    pMainLayout->addWidget(pNewSubnetBtn);
    pMainLayout->setContentsMargins(0, 0, 0, 0);

    setLayout(pMainLayout);

    connect(m_pListView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
}

void ZenoSubnetListPanel::initModel(GraphsModel* pModel)
{
    m_pListView->initModel(pModel);
    m_pTextLbl->setText(pModel->fileName());
}

QSize ZenoSubnetListPanel::sizeHint() const
{
    int w = m_pListView->sizeHint().width();
    int h = QWidget::sizeHint().height();
    return QSize(w, h);
}