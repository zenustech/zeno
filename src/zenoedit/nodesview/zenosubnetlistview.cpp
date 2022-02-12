#include "zenosubnetlistview.h"
#include "zenoapplication.h"
#include "zenosubnettreeview.h"
#include "graphsmanagment.h"
#include "style/zenostyle.h"
#include "../model/graphstreemodel.h"
#include <zenoui/model/graphsmodel.h>
#include "zsubnetlistitemdelegate.h"
#include <comctrl/zlabel.h>
#include <zenoui/model/modelrole.h>
#include <zenoui/include/igraphsmodel.h>


ZSubnetListModel::ZSubnetListModel(IGraphsModel* pModel, QObject* parent)
    : QStandardItemModel(parent)
    , m_model(nullptr)
{
    m_model = qobject_cast<GraphsModel*>(pModel);
    Q_ASSERT(m_model);
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

void ZenoSubnetListView::initModel(IGraphsModel* pModel)
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
    , m_pTreeView(nullptr)
    , m_bListView(true)
{
    QVBoxLayout* pMainLayout = new QVBoxLayout;

    QHBoxLayout* pLabelLayout = new QHBoxLayout;
    QLabel* pIcon = new QLabel;
    pIcon->setPixmap(QIcon(":/icons/ic_File.svg").pixmap(ZenoStyle::dpiScaledSize(QSize(16,16)), QIcon::Normal));

    m_pTextLbl = new QLabel("");
    QFont font = QFont("HarmonyOS Sans", 12);
    font.setBold(false);
    m_pTextLbl->setFont(font);
    QPalette pal = m_pTextLbl->palette();
    pal.setColor(QPalette::WindowText, QColor(128, 124, 122));
    m_pTextLbl->setPalette(pal);

    pLabelLayout->addWidget(pIcon);
    pLabelLayout->addWidget(m_pTextLbl);
    pLabelLayout->addStretch();
    pLabelLayout->setContentsMargins(12, 5, 5, 0);
    pMainLayout->addLayout(pLabelLayout);

    m_pListView = new ZenoSubnetListView;
    m_pTreeView = new ZenoSubnetTreeView;
    pMainLayout->addWidget(m_pListView);
    pMainLayout->addWidget(m_pTreeView);

    ZTextLabel* pNewSubnetBtn = new ZTextLabel("Add New Subnet");
    font.setPointSize(13);
    font.setBold(false);
    pNewSubnetBtn->setFont(font);
    pNewSubnetBtn->setTextColor(QColor(116, 116, 116));
    pNewSubnetBtn->setBackgroundColor(QColor(56, 57, 56));
    pNewSubnetBtn->setAlignment(Qt::AlignCenter);
    pNewSubnetBtn->setFixedHeight(ZenoStyle::dpiScaled(40));

    pMainLayout->addWidget(pNewSubnetBtn);
    pMainLayout->setContentsMargins(0, 0, 0, 0);

    setLayout(pMainLayout);

    connect(m_pListView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
    connect(m_pTreeView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
    connect(pNewSubnetBtn, SIGNAL(clicked()), this, SLOT(onNewSubnetBtnClicked()));
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

    m_pListView->setVisible(m_bListView);
    m_pTreeView->setVisible(!m_bListView);
}

void ZenoSubnetListPanel::initModel(IGraphsModel* pModel)
{
    m_pListView->initModel(pModel);
    //m_pTreeView->initModel(pModel);
    m_pTextLbl->setText(pModel->fileName());
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
}

void ZenoSubnetListPanel::setViewWay(bool bListView)
{
    m_bListView = bListView;
	m_pListView->setVisible(m_bListView);
	m_pTreeView->setVisible(!m_bListView);
}

QSize ZenoSubnetListPanel::sizeHint() const
{
    if (m_bListView)
    {
		int w = m_pListView->sizeHint().width();
		int h = QWidget::sizeHint().height();
		return QSize(w, h);
    }
    else
    {
        int w = m_pTreeView->sizeHint().width();
		int h = QWidget::sizeHint().height();
		return QSize(w, h);
    }
}

void ZenoSubnetListPanel::onModelReset()
{
    hide();
}

void ZenoSubnetListPanel::onNewSubnetBtnClicked()
{
    if (m_bListView)
    {
        //should detach from GraphsModel
		GraphsModel* pModel = qobject_cast<GraphsModel*>(m_pListView->model());
		Q_ASSERT(pModel);
		SubGraphModel* pSubModel = new SubGraphModel(pModel);
		pSubModel->setName("temp");
		pModel->appendSubGraph(pSubModel);

		const QModelIndexList& lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, "temp", 1, Qt::MatchExactly);
		Q_ASSERT(lst.size() == 1);
		const QModelIndex idx = lst[0];
		m_pListView->edit(idx);
    }
}

void ZenoSubnetListPanel::paintEvent(QPaintEvent* e)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(42, 42, 42));
}