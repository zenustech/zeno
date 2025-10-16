#include "zgeometryspreadsheet.h"
#include "../layout/docktabcontent.h"
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/MaterialObject.h>
#include "model/geometrymodel.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "declmetatype.h"
#include "variantptr.h"
#include "zassert.h"
#include "style/zenostyle.h"
#include <zeno/extra/SceneAssembler.h>

static void tableViewResizeColumns(QTableView* tableView) {
    tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    tableView->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    tableView->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    tableView->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    tableView->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);

    // 禁用文本省略号显示，让内容完整显示
    tableView->setTextElideMode(Qt::ElideNone);
    tableView->setEditTriggers(QTableView::NoEditTriggers);  // 完全禁用编辑

    // 禁用点击项时的自动滚动行为
    tableView->setAutoScroll(false);

    // 设置表格大小策略，允许表格超出父组件宽度
    tableView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // 不知为什么右边显示不完整，添加padding
    tableView->setStyleSheet(
        "QTableView::item {"
        "    padding-right: " + QString::number(ZenoStyle::dpiScaled(10))  + "px;"
        "}"
    );

    tableView->setContextMenuPolicy(Qt::CustomContextMenu);
};

void showCopyMenu(QTableView* tableView, const QPoint& pos) {
    // 1. 获取右键点击的模型索引
    QModelIndex index = tableView->indexAt(pos);
    if (!index.isValid()) return; // 点击无效区域直接返回
    // 2. 创建菜单并添加复制动作
    QMenu menu;
    QAction* copyAction = menu.addAction("复制");
    // 3. 显示菜单并等待用户选择
    QAction* selectedAction = menu.exec(tableView->viewport()->mapToGlobal(pos));
    // 4. 处理复制逻辑
    if (selectedAction == copyAction) {
        QString text = index.data(Qt::DisplayRole).toString(); // 获取单元格文本
        QGuiApplication::clipboard()->setText(text);          // 复制到剪贴板
    }
}

void showObjectDialog(QWidget* parent, QWidget* widget, const QString& title)
{
    if (!widget) return;

    // 创建模态对话框
    QDialog dialog(parent);
    dialog.setWindowTitle(title);
    dialog.setMinimumSize(ZenoStyle::dpiScaledSize(QSize(900, 700)));

    // 将现有组件添加到对话框
    widget->setParent(&dialog);
    QVBoxLayout* layout = new QVBoxLayout(&dialog);
    layout->addWidget(widget);
    dialog.setLayout(layout);

    // 显示模态对话框
    dialog.exec();

    // 对话框关闭后，将组件重新设置为nullptr父对象
    widget->setParent(nullptr);
}

static void exportTableViewToCsv(QTableView* tableView) {
    QAbstractItemModel* model = tableView->model();
    if (!model) {
        QMessageBox::warning(nullptr, "错误", "表格没有模型！");
        return;
    }

    int columnCount = model->columnCount();

    // 1. 获取所有列头
    QMap<QString, int> allColumns; // key: header label, value: column index
    for (int col = 0; col < columnCount; ++col) {
        QString header = model->headerData(col, Qt::Horizontal).toString();
        allColumns.insert(header, col);
    }

    // 2. 弹出输入框，获取列名（空格分隔）
    bool ok = false;
    QString input = QInputDialog::getText(
        nullptr,
        "选择导出列",
        QString("请输入列标题，使用空格分隔（可用列：%1）").arg(allColumns.keys().join(" ")),
        QLineEdit::Normal,
        "",
        &ok
    );

    if (!ok)
        return;

    // 3. 解析用户输入，转为列表，并校验列名
    QStringList headersToExport;
    QStringList selectedHeaders = input.split(" ", Qt::SkipEmptyParts);

    if (selectedHeaders.isEmpty()) {
        // 空输入 -> 默认导出所有列
        headersToExport = allColumns.keys();
    }
    else {
        QSet<QString> validHeaders = allColumns.keys().toSet();
        for (const QString& header : selectedHeaders) {
            if (!validHeaders.contains(header)) {
                QMessageBox::warning(nullptr, "错误", QString("列 \"%1\" 不存在！").arg(header));
                return;
            }
            headersToExport << header;
        }
    }

    headersToExport.sort(Qt::CaseInsensitive);  // 按字母排序

    // 4. 选择导出文件路径
    QString filePath = QFileDialog::getSaveFileName(
        nullptr,
        "导出为 CSV",
        "",
        "CSV 文件 (*.csv)"
    );

    if (filePath.isEmpty())
        return;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(nullptr, "错误", "无法打开文件进行写入！");
        return;
    }

    QTextStream out(&file);
    out.setCodec("UTF-8");

    // 5. 写入标题行
    out << headersToExport.join(",") << "\n";

    // 6. 写入数据行
    int rowCount = model->rowCount();
    for (int row = 0; row < rowCount; ++row) {
        QStringList rowData;
        for (const QString& header : headersToExport) {
            int col = allColumns[header];
            QVariant value = model->data(model->index(row, col));
            QString cellData;
            if (value.type() == QMetaType::Double || value.canConvert<double>()) {
                double num = value.toDouble();
                if (abs(num) > 10) {
                    cellData = QString::number(num, 'f', 2);  // 保留4位小数
                }
                else {
                    cellData = QString::number(num, 'f', 4);  // 保留4位小数
                }
            }
            else {
                cellData = value.toString();
            }

            cellData.replace('"', "\"\"");
            if (cellData.contains(',') || cellData.contains('"'))
                cellData = QString("\"%1\"").arg(cellData);
            rowData << cellData;
        }
        out << rowData.join(",") << "\n";
    }

    file.close();
    QMessageBox::information(nullptr, "完成", "导出成功！");
}

UnderlineItemDelegate::UnderlineItemDelegate(QObject* parent) : QStyledItemDelegate(parent) {
}

void UnderlineItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const {
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);

    // 绘制默认背景
    QStyle* style = opt.widget ? opt.widget->style() : QApplication::style();
    style->drawPrimitive(QStyle::PE_PanelItemViewItem, &opt, painter, opt.widget);

    painter->save();

    // 设置文本颜色 - 根据状态变化
    QColor textColor = QColor(100, 180, 255); // 默认浅蓝色

    if (opt.state & QStyle::State_MouseOver || opt.state & QStyle::State_Selected) {
        textColor = Qt::white; // 鼠标悬停或选中时变为白色
    }

    // 设置字体
    QFont font = opt.font;
    painter->setFont(font);
    painter->setPen(textColor);

    // 文本绘制区域（留出边距）
    QRect textRect = opt.rect.adjusted(4, 2, -4, -2);

    // 绘制文本
    painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter, opt.text);

    // 手动绘制下划线（更可靠的方法）
    QFontMetrics fm(font);
    int textWidth = fm.horizontalAdvance(opt.text);
    int underlineY = textRect.bottom() + 1; // 下划线在文本下方

    // 绘制下划线（只覆盖文本宽度）
    QPen underlinePen(textColor);
    underlinePen.setWidth(1);
    painter->setPen(underlinePen);
    painter->drawLine(textRect.left(), underlineY, textRect.left() + textWidth, underlineY);

    painter->restore();
}

ListObjItemDelegate::ListObjItemDelegate(QObject* parent)
    : QStyledItemDelegate(parent)
    , m_underlineDelegate(new UnderlineItemDelegate(this))
{
}

void ListObjItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    // 检查是否为叶子节点（非List项）
    QTreeView* treeView = qobject_cast<QTreeView*>(const_cast<QWidget*>(option.widget));
    if (treeView && treeView->model()) {
        // 如果是叶子节点（没有子项），使用下划线代理
        if (!treeView->model()->hasChildren(index)) {
            m_underlineDelegate->paint(painter, option, index);
            return;
        }
    }

    // 对于有子项的节点（List项），使用默认样式
    QStyledItemDelegate::paint(painter, option, index);
}


// BaseAttributeView组件实现
BaseAttributeView::BaseAttributeView(QWidget* parent)
    : QWidget(parent)
    , m_model(nullptr)
    , m_lblNode(new QLabel)
    , m_vertex(new ZToolBarButton(true, ":/icons/geomsheet_vertex_idle.svg", ":/icons/geomsheet_vertex_on.svg"))
    , m_point(new ZToolBarButton(true, ":/icons/geomsheet_point_idle.svg", ":/icons/geomsheet_point_on.svg"))
    , m_face(new ZToolBarButton(true, ":/icons/geomsheet_face_idle.svg", ":/icons/geomsheet_face_on.svg"))
    , m_geom(new ZToolBarButton(true, ":/icons/geomsheet_geometry_idle.svg", ":/icons/geomsheet_geometry_on.svg"))
    , m_ud(new ZToolBarButton(true, ":/icons/geo_userdata-idle.svg", ":/icons/geo_userdata-on.svg"))
    , m_stackViews(new QStackedWidget(this))
{
    auto vertextTable = new QTableView;
    tableViewResizeColumns(vertextTable);
    auto pointTable = new QTableView;
    tableViewResizeColumns(pointTable);
    auto faceTable = new QTableView;
    tableViewResizeColumns(faceTable);
    auto geoTable = new QTableView;
    tableViewResizeColumns(geoTable);
    auto udTable = new QTableView;
    tableViewResizeColumns(udTable);
    m_stackViews->addWidget(vertextTable); //vertex
    m_stackViews->addWidget(pointTable); //point
    m_stackViews->addWidget(faceTable); //face
    m_stackViews->addWidget(geoTable); //geom
    m_stackViews->addWidget(udTable); //ud

    QLabel* pImgBlank = new QLabel("Current Object Is an image, please watch it in image panel");
    m_stackViews->addWidget(pImgBlank);

    QLabel* pLblBlank = new QLabel("No object available, may be not apply or result is null");
    m_stackViews->addWidget(pLblBlank);    //blank

    QPalette palette = m_lblNode->palette();
    palette.setColor(QPalette::WindowText, Qt::white);  // 设置字体颜色为蓝色
    m_lblNode->setPalette(palette);
    pLblBlank->setPalette(palette);
    pImgBlank->setPalette(palette);

    QHBoxLayout* pToolbarLayout = new QHBoxLayout;
    pToolbarLayout->setContentsMargins(1, 1, 1, 1);  // 去掉工具栏布局边距
    pToolbarLayout->setSpacing(2);  // 设置较小的间距

    QLineEdit* pJumpNum = new QLineEdit;
    pJumpNum->setFixedWidth(56);
    pJumpNum->setStyleSheet("background-color: rgb(61,61,61); border: 0px");
    QPushButton* pJumpBtn = new QPushButton(tr("Jump"));
    pJumpBtn->setProperty("cssClass", "proppanel");
    pJumpBtn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);

    connect(pJumpBtn, &QPushButton::clicked, [=]() {
        bool bOk = false;
        int line = pJumpNum->text().toInt(&bOk);
        if (m_point->isChecked()) {
            QTableView* pTableView = qobject_cast<QTableView*>(m_stackViews->currentWidget());
            QModelIndex index = pTableView->model()->index(line, 0);
            if (index.isValid()) {
                pTableView->scrollTo(index, QAbstractItemView::PositionAtTop);
            }
        }
        });

    QPushButton* pExport = new QPushButton(tr("Export"));
    pExport->setProperty("cssClass", "proppanel");
    pExport->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
    connect(pExport, &QPushButton::clicked, [=]() {
        if (m_point->isChecked()) {
            QTableView* pTableView = qobject_cast<QTableView*>(m_stackViews->currentWidget());
            ZASSERT_EXIT(pTableView);
            exportTableViewToCsv(pTableView);
        }
        });


    pToolbarLayout->addWidget(m_lblNode);

    pToolbarLayout->addWidget(pJumpNum);
    pToolbarLayout->addWidget(pJumpBtn);
    pToolbarLayout->addWidget(pExport);

    pToolbarLayout->addWidget(m_vertex);
    pToolbarLayout->addWidget(m_point);
    pToolbarLayout->addWidget(m_face);
    pToolbarLayout->addWidget(m_geom);
    pToolbarLayout->addWidget(m_ud);

    // 创建一个QWidget容器来包含工具栏布局，并设置背景色
    QWidget* toolbarWidget = new QWidget(this);
    toolbarWidget->setLayout(pToolbarLayout);
    // 设置工具栏背景色为深灰色
    toolbarWidget->setStyleSheet("background-color: rgb(31,31,31);");

    QVBoxLayout* pMainLayout = new QVBoxLayout;
    pMainLayout->setContentsMargins(1, 1, 1, 1);  // 去掉主布局边距
    pMainLayout->setSpacing(2);  // 设置较小的间距
    pMainLayout->addWidget(toolbarWidget);  // 添加工具栏容器而不是直接添加布局
    pMainLayout->addWidget(m_stackViews);

    m_vertex->setChecked(false);
    m_point->setChecked(true);
    m_face->setChecked(false);
    m_geom->setChecked(false);
    m_ud->setChecked(false);

    // 为工具栏按钮添加悬停提示
    m_vertex->setToolTip(tr("vertex"));
    m_point->setToolTip(tr("point"));
    m_face->setToolTip(tr("face"));
    m_geom->setToolTip(tr("geo"));
    m_ud->setToolTip(tr("userData"));

    connect(m_vertex, &ZToolBarButton::toggled, [&](bool bChecked) {
        if (!bChecked) {
            m_vertex->setChecked(true);
            return;
        }
        m_point->setChecked(!bChecked);
        m_face->setChecked(!bChecked);
        m_geom->setChecked(!bChecked);
        m_ud->setChecked(!bChecked);
        m_stackViews->setCurrentIndex(0);
        });

    connect(m_point, &ZToolBarButton::toggled, [&](bool bChecked) {
        if (!bChecked) {
            m_point->setChecked(true);
            return;
        }
        m_vertex->setChecked(!bChecked);
        m_face->setChecked(!bChecked);
        m_geom->setChecked(!bChecked);
        m_ud->setChecked(!bChecked);
        m_stackViews->setCurrentIndex(1);
        });

    connect(m_face, &ZToolBarButton::toggled, [&](bool bChecked) {
        if (!bChecked) {
            m_face->setChecked(true);
            return;
        }
        m_vertex->setChecked(!bChecked);
        m_point->setChecked(!bChecked);
        m_geom->setChecked(!bChecked);
        m_ud->setChecked(!bChecked);
        m_stackViews->setCurrentIndex(2);
        });

    connect(m_geom, &ZToolBarButton::toggled, [&](bool bChecked) {
        if (!bChecked) {
            m_geom->setChecked(true);
            return;
        }
        m_vertex->setChecked(!bChecked);
        m_face->setChecked(!bChecked);
        m_point->setChecked(!bChecked);
        m_ud->setChecked(!bChecked);
        m_stackViews->setCurrentIndex(3);
        });

    connect(m_ud, &ZToolBarButton::toggled, [&](bool bChecked) {
        if (!bChecked) {
            m_ud->setChecked(true);
            return;
        }
        m_vertex->setChecked(!bChecked);
        m_face->setChecked(!bChecked);
        m_point->setChecked(!bChecked);
        m_geom->setChecked(!bChecked);
        m_stackViews->setCurrentIndex(4);
        });

    connect(vertextTable, &QTableView::customContextMenuRequested, this, [vertextTable](const QPoint& pos) { showCopyMenu(vertextTable, pos); });
    connect(pointTable, &QTableView::customContextMenuRequested, this, [pointTable](const QPoint& pos) { showCopyMenu(pointTable, pos); });
    connect(faceTable, &QTableView::customContextMenuRequested, this, [faceTable](const QPoint& pos) { showCopyMenu(faceTable, pos); });
    connect(geoTable, &QTableView::customContextMenuRequested, this, [geoTable](const QPoint& pos) { showCopyMenu(geoTable, pos); });
    connect(udTable, &QTableView::customContextMenuRequested, this, [udTable](const QPoint& pos) { showCopyMenu(udTable, pos); });

    setLayout(pMainLayout);
}

void BaseAttributeView::setGeometryObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::GeometryObject_Adapter* object, QString nodeName)
{
    if (nodeidx.isValid()) {
        m_nodeIdx = nodeidx;
    }
    m_lblNode->setText(nodeName);

    if (subgraph) {
        m_model = subgraph;
        connect(m_model, &GraphModel::nodeRemoved, this, &BaseAttributeView::onNodeRemoved, Qt::UniqueConnection);
        connect(m_model, &GraphModel::dataChanged, this, &BaseAttributeView::onNodeDataChanged, Qt::UniqueConnection);
    }

    if (m_vertex->isChecked())
        m_stackViews->setCurrentIndex(0);

    if (m_point->isChecked())
        m_stackViews->setCurrentIndex(1);

    if (m_face->isChecked())
        m_stackViews->setCurrentIndex(2);

    if (m_geom->isChecked())
        m_stackViews->setCurrentIndex(3);

    if (m_ud->isChecked())
        m_stackViews->setCurrentIndex(4);

    // 设置各个视图的模型
    QTableView* view = qobject_cast<QTableView*>(m_stackViews->widget(0));
    if (VertexModel* model = qobject_cast<VertexModel*>(view->model())) {
        model->setGeoObject(object);
    } else {
        view->setModel(new VertexModel(object));
    }

    view = qobject_cast<QTableView*>(m_stackViews->widget(1));
    if (PointModel* model = qobject_cast<PointModel*>(view->model())) {
        model->setGeoObject(object);
    }
    else {
        view->setModel(new PointModel(object));
    }

    view = qobject_cast<QTableView*>(m_stackViews->widget(2));
    if (FaceModel* model = qobject_cast<FaceModel*>(view->model())) {
        model->setGeoObject(object);
    }
    else {
        view->setModel(new FaceModel(object));
    }

    view = qobject_cast<QTableView*>(m_stackViews->widget(3));
    if (GeomDetailModel* model = qobject_cast<GeomDetailModel*>(view->model())) {
        model->setGeoObject(object);
    }
    else {
        view->setModel(new GeomDetailModel(object));
    }

    view = qobject_cast<QTableView*>(m_stackViews->widget(4));
    if (GeomUserDataModel* udmodel = qobject_cast<GeomUserDataModel*>(view->model())) {
        udmodel->setGeoObject(object);
    }
    else {
        view->setModel(new GeomUserDataModel(object));
    }
}

void BaseAttributeView::clearModel()
{
    for (int i = 0; i < m_stackViews->count(); ++i) {
        QTableView* view = qobject_cast<QTableView*>(m_stackViews->widget(i));
        if (!view) continue;
        if (QAbstractItemModel* model = view->model()) {
            view->setModel(nullptr);
            delete model;
        }
    }
    m_model = nullptr;
    m_nodeIdx = QPersistentModelIndex();
    m_lblNode->setText("");
}

void BaseAttributeView::onNodeRemoved(QString nodename)
{
    if (!m_nodeIdx.isValid()) {
        clearModel();
    }
}

void BaseAttributeView::onNodeDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (topLeft.data(QtRole::ROLE_NODE_NAME).toString() == m_nodeIdx.data(QtRole::ROLE_NODE_NAME).toString()) {
        if (!roles.empty() && roles[0] == QtRole::ROLE_NODE_RUN_STATE) {
            QmlNodeRunStatus::Value currStatus = topLeft.data(QtRole::ROLE_NODE_RUN_STATE).value<QmlNodeRunStatus::Value>();
            if (currStatus == zeno::Node_Running) {
                clearModel();
            }
            else if (currStatus == zeno::Node_RunSucceed) {
                auto pObject = m_nodeIdx.data(QtRole::ROLE_OUTPUT_OBJS).value<zeno::IObject*>();

                if (!pObject) {
                    m_stackViews->setCurrentIndex(m_stackViews->count() - 1);
                    return;
                } else {
                    if (pObject->userData()->has("isImage")) {
                        m_stackViews->setCurrentIndex(m_stackViews->count() - 2);
                        return;
                    }
                    if (auto spGeom = dynamic_cast<zeno::GeometryObject_Adapter*>(pObject)) {
                        setGeometryObject(QVariantPtr<GraphModel>::asPtr(m_nodeIdx.data(QtRole::ROLE_GRAPH)), m_nodeIdx, spGeom, m_nodeIdx.data(QtRole::ROLE_NODE_NAME).toString());
                    }
                }
            }
        }
    }
}

// SceneTreeNodeWidget实现 - 调整：meshes和children放在同一个tableview中，两列显示
SceneTreeNodeWidget::SceneTreeNodeWidget(QWidget* parent)
    : QWidget(parent)  // 修改：移除Qt::Tool | Qt::FramelessWindowHint标志
    , m_dataModel(new QStandardItemModel(this))
{
    // 设置背景色为浅灰色，边框为2像素黑色实线
    setStyleSheet("SceneTreeNodeWidget {background-color: black}");
    setMinimumSize(ZenoStyle::dpiScaled(500), ZenoStyle::dpiScaled(300));

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(2, 2, 2, 2);
    layout->setSpacing(0);

    // 创建一个QWidget来包含三个标签
    QWidget* labelsWidget = new QWidget(this);
    labelsWidget->setStyleSheet("background-color: rgb(31,31,31); color: #A3B1C0;");
    QVBoxLayout* labelsLayout = new QVBoxLayout(labelsWidget);
    labelsLayout->setContentsMargins(2, 2, 2, 2);
    labelsLayout->setSpacing(2);

    // matrix和visibility用label显示
    m_matrixLabel = new QLabel("Matrix: ");
    m_visibilityLabel = new QLabel("Visibility: ");
    QLabel* meshesLabel = new QLabel("Meshes and Children:");

    // 将三个标签添加到labelsLayout中
    labelsLayout->addWidget(m_matrixLabel);
    labelsLayout->addWidget(m_visibilityLabel);
    labelsLayout->addWidget(meshesLabel);

    // 将labelsWidget添加到主布局
    layout->addWidget(labelsWidget);

    // meshes和children用同一个tableview显示，两列
    m_dataTableView = new QTableView;
    m_dataTableView->setModel(m_dataModel);
    layout->addWidget(m_dataTableView);

    tableViewResizeColumns(m_dataTableView);
    connect(m_dataTableView, &QTableView::customContextMenuRequested, this, [this](const QPoint& pos) { showCopyMenu(m_dataTableView, pos); });
}

void SceneTreeNodeWidget::setTreeNode(const zeno::SceneTreeNode& treeNode)
{
    m_matrixLabel->setText(QString("Matrix: %1").arg(QString::fromStdString(treeNode.matrix)));
    m_visibilityLabel->setText(QString("Visibility: %1").arg(treeNode.visibility ? "true" : "false"));

    // 清空现有数据，重用model
    m_dataModel->clear();
    m_dataModel->setColumnCount(2);
    m_dataModel->setHorizontalHeaderLabels({ "Meshes", "Children" });

    // 计算最大行数
    int maxRows = std::max(treeNode.meshes.size(), treeNode.children.size());

    // 填充表格数据
    for (int row = 0; row < maxRows; ++row) {
        // 添加meshes列数据
        if (row < treeNode.meshes.size()) {
            m_dataModel->setItem(row, 0, new QStandardItem(QString::fromStdString(treeNode.meshes[row])));
        }
        else {
            m_dataModel->setItem(row, 0, new QStandardItem(""));  // 空单元格
        }

        // 添加children列数据
        if (row < treeNode.children.size()) {
            m_dataModel->setItem(row, 1, new QStandardItem(QString::fromStdString(treeNode.children[row])));
        }
        else {
            m_dataModel->setItem(row, 1, new QStandardItem(""));  // 空单元格
        }
    }
}

void SceneTreeNodeWidget::clearModel()
{
    m_dataModel->clear();

    m_matrixLabel->setText("Matrix: ");
    m_visibilityLabel->setText("Visibility: ");
}

// MatrixWidget实现
MatrixWidget::MatrixWidget(QWidget* parent)
    : QWidget(parent)  // 修改：移除Qt::Tool | Qt::FramelessWindowHint标志
    , m_matrixModel(new QStandardItemModel(this))
{
    // 设置背景色为浅灰色，边框为2像素黑色实线
    setStyleSheet("MatrixWidget {background-color: black}");
    setMinimumSize(ZenoStyle::dpiScaled(500), ZenoStyle::dpiScaled(300));

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(2, 2, 2, 2);

    m_matrixTableView = new QTableView;
    m_matrixTableView->setModel(m_matrixModel);
    layout->addWidget(m_matrixTableView);

    tableViewResizeColumns(m_matrixTableView);
    connect(m_matrixTableView, &QTableView::customContextMenuRequested, this, [this](const QPoint& pos) { showCopyMenu(m_matrixTableView, pos); });
}

void MatrixWidget::setMatrices(const std::vector<glm::mat4>& matrices)
{
    // 清空现有数据，重用model
    m_matrixModel->clear();
    m_matrixModel->setColumnCount(4);
    m_matrixModel->setHorizontalHeaderLabels({ "X", "Y", "Z", "W" });

    for (size_t i = 0; i < matrices.size(); ++i) {
        const auto& mat = matrices[i];
        QList<QStandardItem*> row;

        // 显示矩阵的每一行（4x4矩阵，但我们只显示4列）
        for (int col = 0; col < 4; ++col) {
            QString value;
            for (int rowIdx = 0; rowIdx < 4; ++rowIdx) {
                if (rowIdx > 0) value += ", ";
                value += QString::number(mat[col][rowIdx], 'f', 6);
            }
            row << new QStandardItem(value);
        }
        m_matrixModel->appendRow(row);
    }
}

void MatrixWidget::clearModel()
{
    m_matrixModel->clear();
}

// SceneObjView组件实现
SceneObjView::SceneObjView(QWidget* parent)
    : QWidget(parent)
    , m_lblNode(new QLabel)
    , m_radioSceneTree(new QRadioButton("scene_tree"))  // 修改：改为QRadioButton
    , m_radioNodeToMatrix(new QRadioButton("node_to_matrix"))  // 修改：改为QRadioButton
    , m_radioNodeToId(new QRadioButton("node_to_id"))  // 修改：改为QRadioButton
    , m_radioGeomList(new QRadioButton("geom_list"))  // 修改：改为QRadioButton
    , m_stackViews(new QStackedWidget(this))
    , m_sceneTreeNodeWidget(new SceneTreeNodeWidget(nullptr))
    , m_matrixWidget(new MatrixWidget(nullptr))
    , m_baseAttributeView(new BaseAttributeView(nullptr))  // 修改：直接使用BaseAttributeView替换GeometryWidget
{
    // 创建自定义委托实例
    UnderlineItemDelegate* underlineDelegate = new UnderlineItemDelegate(this);

    // 创建3个QListView用于显示SceneObjectListModel数据
    for (int i = 0; i < 3; i++) {
        QListView* listView = new QListView();
        listView->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        listView->setTextElideMode(Qt::ElideRight);
        listView->setWordWrap(false);

        listView->setItemDelegate(underlineDelegate);
        listView->setCursor(Qt::PointingHandCursor);

        m_stackViews->addWidget(listView);
    }
    QTableView* tableView = new QTableView;
    tableViewResizeColumns(tableView);
    connect(tableView, &QTableView::customContextMenuRequested, this, [tableView](const QPoint& pos) { showCopyMenu(tableView, pos); });

    m_stackViews->addWidget(tableView);

    // 设置单选按钮互斥行为
    m_radioSceneTree->setChecked(true); // 默认选中第一个
    QButtonGroup* buttonGroup = new QButtonGroup(this);
    buttonGroup->setExclusive(true);
    buttonGroup->addButton(m_radioSceneTree);
    buttonGroup->addButton(m_radioNodeToMatrix);
    buttonGroup->addButton(m_radioGeomList);
    buttonGroup->addButton(m_radioNodeToId);

    // 连接单选按钮切换事件
    connect(m_radioSceneTree, &QRadioButton::toggled, this, &SceneObjView::onRadioButtonToggled);
    connect(m_radioNodeToMatrix, &QRadioButton::toggled, this, &SceneObjView::onRadioButtonToggled);
    connect(m_radioNodeToId, &QRadioButton::toggled, this, &SceneObjView::onRadioButtonToggled);
    connect(m_radioGeomList, &QRadioButton::toggled, this, &SceneObjView::onRadioButtonToggled);

    // 设置标签样式
    QPalette palette = m_lblNode->palette();
    palette.setColor(QPalette::WindowText, Qt::white);
    m_lblNode->setPalette(palette);

    // 设置单选按钮文本颜色与标签保持一致
    QPalette radioPalette = m_radioSceneTree->palette();
    radioPalette.setColor(QPalette::WindowText, Qt::white);
    radioPalette.setColor(QPalette::Text, Qt::white);
    m_radioSceneTree->setPalette(radioPalette);
    m_radioNodeToMatrix->setPalette(radioPalette);
    m_radioNodeToId->setPalette(radioPalette);
    m_radioGeomList->setPalette(radioPalette);

    // 创建水平布局用于标签和单选按钮
    QHBoxLayout* pTopLayout = new QHBoxLayout;
    pTopLayout->setContentsMargins(1, 1, 1, 1);
    pTopLayout->setSpacing(2);
    pTopLayout->addWidget(m_lblNode);
    pTopLayout->addStretch();
    pTopLayout->addWidget(m_radioSceneTree);
    pTopLayout->addWidget(m_radioNodeToMatrix);
    pTopLayout->addWidget(m_radioGeomList);
    pTopLayout->addWidget(m_radioNodeToId);

    // 创建主垂直布局
    QVBoxLayout* pMainLayout = new QVBoxLayout;
    pMainLayout->setContentsMargins(1, 1, 1, 1);
    pMainLayout->setSpacing(2);
    pMainLayout->addLayout(pTopLayout);
    pMainLayout->addWidget(m_stackViews);

    setLayout(pMainLayout);
}

void SceneObjView::onRadioButtonToggled(bool checked)  // 修改：改为单选按钮切换事件
{
    if (!checked) return;

    QRadioButton* senderRadio = qobject_cast<QRadioButton*>(sender());
    if (!senderRadio) return;

    if (senderRadio == m_radioSceneTree) {
        m_stackViews->setCurrentIndex(0);
    }
    else if (senderRadio == m_radioNodeToMatrix) {
        m_stackViews->setCurrentIndex(1);
    }
    else if (senderRadio == m_radioGeomList) {
        m_stackViews->setCurrentIndex(2);
    }
    else if (senderRadio == m_radioNodeToId) {
        m_stackViews->setCurrentIndex(3);
    }
}

void SceneObjView::onSceneTreeClicked(const QModelIndex& index)
{
    if (!index.isValid()) return;

    // 获取当前模型
    QListView* listView = qobject_cast<QListView*>(sender());
    if (!listView) return;

    SceneObjectListModel* model = qobject_cast<SceneObjectListModel*>(listView->model());
    if (!model) return;

    // 获取对应的SceneTreeNode
    zeno::SceneTreeNode treeNode = model->getTreeNodeAt(index.row());

    if (m_sceneTreeNodeWidget) {
        m_sceneTreeNodeWidget->setTreeNode(treeNode);
        // 使用公共函数显示模态对话框
        showObjectDialog(this, m_sceneTreeNodeWidget, "Scene Tree Node Details");
    }
}

void SceneObjView::onNodeToMatrixClicked(const QModelIndex& index)
{
    if (!index.isValid()) return;

    // 获取当前模型
    QListView* listView = qobject_cast<QListView*>(sender());
    if (!listView) return;

    SceneObjectListModel* model = qobject_cast<SceneObjectListModel*>(listView->model());
    if (!model) return;

    // 获取对应的矩阵数据
    std::string key = model->getKeyAt(index.row());

    // 从SceneObject中获取矩阵数据
    if (auto sceneObject = model->getSceneObject()) {
        auto it = sceneObject->node_to_matrix.find(key);
        if (it != sceneObject->node_to_matrix.end()) {
            // 复用现有的m_matrixWidget成员变量
            if (m_matrixWidget) {
                m_matrixWidget->setMatrices(it->second);
                showObjectDialog(this, m_matrixWidget, "Matrix Details");
            }
        }
    }
}

void SceneObjView::onGeomListClicked(const QModelIndex& index)
{
    if (!index.isValid()) return;

    // 获取当前模型
    QListView* listView = qobject_cast<QListView*>(sender());
    if (!listView) return;

    SceneObjectListModel* model = qobject_cast<SceneObjectListModel*>(listView->model());
    if (!model) return;

    // 获取对应的几何体数据
    std::string key = model->getKeyAt(index.row());

    // 从SceneObject中获取几何体数据
    if (auto sceneObject = model->getSceneObject()) {
        auto it = sceneObject->geom_list.find(key);
        if (it != sceneObject->geom_list.end()) {
            // 直接使用BaseAttributeView成员变量
            if (m_baseAttributeView) {
                m_baseAttributeView->setGeometryObject(nullptr, QModelIndex(), it->second.get(), QString::fromStdString(key));
                showObjectDialog(this, m_baseAttributeView, "Geometry Details");
            }
        }
    }
}

void SceneObjView::setSceneObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::SceneObject* pObject, QString nodename)
{
    if (nodeidx.isValid()) {
        m_nodeIdx = nodeidx;
    }
    m_lblNode->setText(nodename);

    if (subgraph) {
        m_model = subgraph;
        connect(m_model, &GraphModel::nodeRemoved, this, &SceneObjView::onNodeRemoved, Qt::UniqueConnection);
    }

    // scene_tree使用QListView + SceneObjectListModel
    QListView* listView = qobject_cast<QListView*>(m_stackViews->widget(0));
    if (SceneObjectListModel* model = qobject_cast<SceneObjectListModel*>(listView->model())) {
        model->setSceneObject(pObject);
    } else {
        auto modelSt = new SceneObjectListModel(SceneObjectListModel::SceneTree, this);
        modelSt->setSceneObject(pObject);
        listView->setModel(modelSt);
    }
    // 连接点击事件
    connect(listView, &QListView::clicked, this, &SceneObjView::onSceneTreeClicked, Qt::UniqueConnection);

    // node_to_matrix使用QListView + SceneObjectListModel
    listView = qobject_cast<QListView*>(m_stackViews->widget(1));
    if (SceneObjectListModel* model = qobject_cast<SceneObjectListModel*>(listView->model())) {
        model->setSceneObject(pObject);
    } else {
        auto modelMatrix = new SceneObjectListModel(SceneObjectListModel::NodeToMatrix, this);
        modelMatrix->setSceneObject(pObject);
        listView->setModel(modelMatrix);
    }
    // 连接点击事件
    connect(listView, &QListView::clicked, this, &SceneObjView::onNodeToMatrixClicked, Qt::UniqueConnection);

    // geom_list使用QListView + SceneObjectListModel
    listView = qobject_cast<QListView*>(m_stackViews->widget(2));
    if (SceneObjectListModel* model = qobject_cast<SceneObjectListModel*>(listView->model())) {
        model->setSceneObject(pObject);
    } else {
        auto modelGeo = new SceneObjectListModel(SceneObjectListModel::GeometryList, this);
        modelGeo->setSceneObject(pObject);
        listView->setModel(modelGeo);
    }
    // 连接点击事件
    connect(listView, &QListView::clicked, this, &SceneObjView::onGeomListClicked, Qt::UniqueConnection);

    // node_to_id使用QTableView + SceneObjectTableModel（修复构造函数调用）
    QTableView* tableView = qobject_cast<QTableView*>(m_stackViews->widget(3));
    if (SceneObjectTableModel* model = qobject_cast<SceneObjectTableModel*>(tableView->model())) {
        model->setSceneObject(pObject);
    } else {
        auto modelNodeToId = new SceneObjectTableModel(SceneObjectTableModel::NodeToId, this);
        modelNodeToId->setSceneObject(pObject);
        tableView->setModel(modelNodeToId);
    }

    // 根据当前选中的单选按钮设置初始视图
    if (m_radioSceneTree->isChecked()) {
        m_stackViews->setCurrentIndex(0);
    } else if (m_radioNodeToMatrix->isChecked()) {
        m_stackViews->setCurrentIndex(1);
    } else if (m_radioGeomList->isChecked()) {
        m_stackViews->setCurrentIndex(2);
    } else if (m_radioNodeToId->isChecked()) {
        m_stackViews->setCurrentIndex(3);
    }
}

void SceneObjView::clearModel()
{
    // 清理所有stacked widget中的视图模型
    for (int i = 0; i < m_stackViews->count(); ++i) {
        QAbstractItemView* view = qobject_cast<QAbstractItemView*>(m_stackViews->widget(i));
        if (!view) continue;
        view->disconnect();
        if (QAbstractItemModel* model = view->model()) {
            view->setModel(nullptr);
            delete model;
        }
    }

    if (m_sceneTreeNodeWidget) {
        m_sceneTreeNodeWidget->clearModel();
    }
    if (m_matrixWidget) {
        m_matrixWidget->clearModel();
    }
    if (m_baseAttributeView) {
        m_baseAttributeView->clearModel();
    }

    m_model = nullptr;
    m_nodeIdx = QPersistentModelIndex();
    m_lblNode->setText("");
}

void SceneObjView::onNodeRemoved(QString nodename)
{
    if (!m_nodeIdx.isValid()) {
        clearModel();
    }
}

// MaterialObjView实现
MaterialObjView::MaterialObjView(QWidget* parent)
    : QWidget(parent)
    , m_lblNode(new QLabel)
    , m_materialTableView(new QTableView)
    , m_materialModel(new QStandardItemModel(this))
    , m_model(nullptr)
{
    // 设置节点标签样式
    QPalette palette = m_lblNode->palette();
    palette.setColor(QPalette::WindowText, Qt::white);
    m_lblNode->setPalette(palette);

    // 设置表格模型 - 两列：第一列是key，第二列是value
    m_materialModel->setColumnCount(2);
    m_materialModel->setHorizontalHeaderLabels({ "Property", "Value" });
    m_materialTableView->setModel(m_materialModel);

    // 设置表格样式
    tableViewResizeColumns(m_materialTableView);
    connect(m_materialTableView, &QTableView::customContextMenuRequested, this, [this](const QPoint& pos) { showCopyMenu(m_materialTableView, pos); });
    m_materialTableView->setWordWrap(true);

    // 设置行高自适应，确保长文本能够完整显示
    m_materialTableView->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);

    // 设置主布局
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(1, 1, 1, 1);
    mainLayout->setSpacing(2);
    mainLayout->addWidget(m_lblNode);
    mainLayout->addWidget(m_materialTableView);

    setLayout(mainLayout);
}

void MaterialObjView::setMaterialObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::MaterialObject* pObject, QString nodename)
{
    if (nodeidx.isValid()) {
        m_nodeIdx = nodeidx;
    }
    m_lblNode->setText(nodename);

    if (subgraph) {
        m_model = subgraph;
        connect(m_model, &GraphModel::nodeRemoved, this, &MaterialObjView::onNodeRemoved, Qt::UniqueConnection);
    }

    // 清空现有数据
    m_materialModel->removeRows(0, m_materialModel->rowCount());

    if (pObject) {
        // 辅助函数：创建并设置value列的项
        auto addProperty = [this](int& row, const QString& propertyName, const QString& value) {
            m_materialModel->setItem(row, 0, new QStandardItem(propertyName));
            QStandardItem* valueItem = new QStandardItem(value);
            valueItem->setTextAlignment(Qt::AlignTop | Qt::AlignLeft);
            m_materialModel->setItem(row, 1, valueItem);
            row++;
            };

        int row = 0;

        // 添加基本属性
        addProperty(row, "vert", QString::fromStdString(pObject->vert));
        addProperty(row, "frag", QString::fromStdString(pObject->frag));
        addProperty(row, "common", QString::fromStdString(pObject->common));
        addProperty(row, "extensions", QString::fromStdString(pObject->extensions));
        addProperty(row, "parameters", QString::fromStdString(pObject->parameters));
        addProperty(row, "mtlidkey", QString::fromStdString(pObject->mtlidkey));

        // 添加tex2Ds属性 - 拼接所有Texture2DObject的path
        QStringList tex2DPaths;
        for (const auto& tex2D : pObject->tex2Ds) {
            if (tex2D) {
                tex2DPaths << QString::fromStdString(tex2D->path);
            }
        }
        addProperty(row, "tex2Ds", tex2DPaths.join("\n"));

        // 添加tex3Ds属性 - 拼接所有TextureObjectVDB的path和channel
        QStringList tex3DInfos;
        for (const auto& tex3D : pObject->tex3Ds) {
            if (tex3D) {
                QString info = QString::fromStdString(tex3D->path);
                if (!tex3D->channel.empty()) {
                    info += " (channel: " + QString::fromStdString(tex3D->channel) + ")";
                }
                tex3DInfos << info;
            }
        }
        addProperty(row, "tex3Ds", tex3DInfos.join("\n"));
    }
}

void MaterialObjView::clearModel()
{
    m_materialModel->removeRows(0, m_materialModel->rowCount());
    m_lblNode->setText("");
    m_model = nullptr;
    m_nodeIdx = QPersistentModelIndex();
}

void MaterialObjView::onNodeRemoved(QString nodename)
{
    if (!m_nodeIdx.isValid()) {
        clearModel();
    }
}

// ListObjView实现
ListObjView::ListObjView(QWidget* parent)
    : QWidget(parent)
    , m_lblNode(new QLabel)
    , m_expandCheckBox(new QCheckBox("展开所有"))
    , m_listTreeView(new QTreeView)
    , m_listModel(new QStandardItemModel(this))
    , m_underlineDelegate(new ListObjItemDelegate(this))
    , m_model(nullptr)
    , m_currentListObject(nullptr)
    , m_baseAttributeView(new BaseAttributeView(nullptr))
    , m_sceneObjView(new SceneObjView(nullptr))
    , m_materialObjView(new MaterialObjView(nullptr))
{
    // 设置节点标签样式
    QPalette palette = m_lblNode->palette();
    palette.setColor(QPalette::WindowText, Qt::white);
    m_lblNode->setPalette(palette);

    // 设置展开复选框样式
    m_expandCheckBox->setChecked(true); // 默认展开
    m_expandCheckBox->setStyleSheet("QCheckBox { color: white; }");

    // 设置树形模型
    m_listModel->setColumnCount(1);
    m_listModel->setHorizontalHeaderLabels({"Object Key"});
    m_listTreeView->setModel(m_listModel);

    // 设置树形视图样式
    m_listTreeView->setHeaderHidden(false);
    m_listTreeView->setAlternatingRowColors(true);
    m_listTreeView->setAnimated(true);
    m_listTreeView->setIndentation(20);
    m_listTreeView->setSortingEnabled(false);

    // 设置文本不省略，允许换行
    m_listTreeView->setTextElideMode(Qt::ElideNone);
    m_listTreeView->setWordWrap(true);
    m_listTreeView->setEditTriggers(QAbstractItemView::NoEditTriggers);

    // 设置自定义委托 - 非List项使用UnderlineItemDelegate
    m_listTreeView->setItemDelegate(m_underlineDelegate);
    m_listTreeView->setCursor(Qt::PointingHandCursor);

    // 设置上下文菜单
    m_listTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_listTreeView, &QTreeView::customContextMenuRequested, this, [this](const QPoint& pos) {
        QModelIndex index = m_listTreeView->indexAt(pos);
        if (!index.isValid()) return;
        QMenu menu;
        QAction* copyAction = menu.addAction("复制");
        QAction* selectedAction = menu.exec(m_listTreeView->viewport()->mapToGlobal(pos));
        if (selectedAction == copyAction) {
            QString text = index.data(Qt::DisplayRole).toString(); // 获取单元格文本
            QGuiApplication::clipboard()->setText(text);          // 复制到剪贴板
        }
    });

    // 连接点击事件
    connect(m_listTreeView, &QTreeView::clicked, this, &ListObjView::onItemClicked);

    // 连接展开复选框信号
    connect(m_expandCheckBox, &QCheckBox::toggled, this, &ListObjView::onExpandAllToggled);

    // 创建工具栏布局
    QHBoxLayout* toolbarLayout = new QHBoxLayout;
    toolbarLayout->setContentsMargins(1, 1, 1, 1);
    toolbarLayout->setSpacing(10);
    toolbarLayout->addWidget(m_lblNode);
    toolbarLayout->addStretch();
    toolbarLayout->addWidget(m_expandCheckBox);

    // 设置主布局
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(1, 1, 1, 1);
    mainLayout->setSpacing(2);
    mainLayout->addLayout(toolbarLayout);
    mainLayout->addWidget(m_listTreeView);

    setLayout(mainLayout);
}

void ListObjView::setListObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::ListObject* pObject)
{
    if (nodeidx.isValid()) {
        QString nodename = nodeidx.data(QtRole::ROLE_NODE_NAME).toString();
        m_lblNode->setText(nodename);
        m_nodeIdx = nodeidx;
        m_currentListObject = pObject;
    }

    if (subgraph) {
        m_model = subgraph;
        connect(m_model, &GraphModel::nodeRemoved, this, &ListObjView::onNodeRemoved, Qt::UniqueConnection);
    }

    // 清空现有数据
    m_listModel->removeRows(0, m_listModel->rowCount());
    
    if (pObject) {
        // 递归添加ListObject的所有项（从空索引路径开始）
        addListObjectItems(nullptr, pObject, QVector<int>(), "root");

        // 根据复选框状态展开或收缩
        if (m_expandCheckBox && m_expandCheckBox->isChecked()) {
            m_listTreeView->expandAll();
        } else {
            m_listTreeView->collapseAll();
        }
    }
}

void ListObjView::addListObjectItems(QStandardItem* parentItem, zeno::ListObject* listObj, QVector<int> parentIndices, const QString& parentName)
{
    if (!listObj) return;

    for (size_t i = 0; i < listObj->size(); ++i) {
        if (auto obj = listObj->get(i)) {
            QString keyName = obj->m_key.empty() ? QString("Item %1").arg(i) : QString::fromStdString(zsString2Std(obj->m_key));

            // 构建当前项的索引路径
            QVector<int> currentIndices = parentIndices;
            currentIndices.append(i);

            // 检查对象是否为ListObject
            if (auto nestedListObj = dynamic_cast<zeno::ListObject*>(obj)) {
                // 如果是ListObject，创建父节点并递归添加子项
                QStandardItem* listItem = new QStandardItem(keyName + " (List)");
                // 将索引路径存储在item的user data中
                listItem->setData(QVariant::fromValue(currentIndices), Qt::UserRole);
                if (parentItem) {
                    parentItem->appendRow(listItem);
                } else {
                    m_listModel->appendRow(listItem);
                }
                // 递归添加嵌套的ListObject项
                addListObjectItems(listItem, nestedListObj, currentIndices, keyName);
            } else {
                // 如果不是ListObject，直接显示key
                QStandardItem* keyItem = new QStandardItem(keyName);
                // 将索引路径存储在item的user data中
                keyItem->setData(QVariant::fromValue(currentIndices), Qt::UserRole);
                if (parentItem) {
                    parentItem->appendRow(keyItem);
                } else {
                    m_listModel->appendRow(keyItem);
                }
            }
        }
    }
}

void ListObjView::clearModel()
{
    m_listModel->removeRows(0, m_listModel->rowCount());
    m_lblNode->setText("");
    m_model = nullptr;
    m_nodeIdx = QPersistentModelIndex();

    m_baseAttributeView->clearModel();
    m_sceneObjView->clearModel();
    m_materialObjView->clearModel();
}

void ListObjView::onNodeRemoved(QString nodename)
{
    if (!m_nodeIdx.isValid()) {
        clearModel();
    }
}

void ListObjView::onExpandAllToggled(bool expanded)
{
    if (expanded) {
        m_listTreeView->expandAll();
    } else {
        m_listTreeView->collapseAll();
    }
}

void ListObjView::onItemClicked(const QModelIndex& index)
{
    if (!index.isValid()) return;

    // 检查是否为非List项（没有子项的叶子节点）
    QStandardItem* item = m_listModel->itemFromIndex(index);
    if (!item || item->hasChildren()) return;

    // 从item的user data中获取索引路径
    QVariant indexVariant = item->data(Qt::UserRole);
    if (!indexVariant.isValid() || !m_currentListObject) return;

    QVector<int> indices = indexVariant.value<QVector<int>>();
    if (indices.isEmpty()) return;

    // 根据索引路径从根对象中重新查找对象
    zeno::IObject* currentObj = m_currentListObject;
    for (int i = 0; i < indices.size() && currentObj; i++) {
        int idx = indices[i];

        if (auto listObj = dynamic_cast<zeno::ListObject*>(currentObj)) {
            if (idx >= 0 && idx < listObj->size()) {
                currentObj = listObj->get(idx);
            } else {
                return;
            }
        } else {
            return;
        }
    }

    if (currentObj) {
        if (auto sceneObj = dynamic_cast<zeno::SceneObject*>(currentObj)) {
            if (m_sceneObjView) {
                m_sceneObjView->setSceneObject(m_model, m_nodeIdx, sceneObj, item->text());
                showObjectDialog(this, m_sceneObjView, "Scene Object Details");
            }
        } else if (auto materialObj = dynamic_cast<zeno::MaterialObject*>(currentObj)) {
            if (m_materialObjView) {
                m_materialObjView->setMaterialObject(m_model, m_nodeIdx, materialObj, item->text());
                showObjectDialog(this, m_materialObjView, "Material Object Details");
            }
        } else {
            zeno::GeometryObject_Adapter* curObj;
            if (auto geoObj = dynamic_cast<zeno::GeometryObject_Adapter*>(currentObj)) {
                curObj = geoObj;
                if (m_baseAttributeView) {
                    m_baseAttributeView->setGeometryObject(m_model, QModelIndex(), curObj, item->text());
                    showObjectDialog(this, m_baseAttributeView, "Geometry Object Details");
                }
            } else if (auto primitiveObj = dynamic_cast<zeno::PrimitiveObject*>(currentObj)) {
                auto curObj = create_GeometryObject(primitiveObj);
                if (m_baseAttributeView) {
                    m_baseAttributeView->setGeometryObject(m_model, QModelIndex(), curObj.get(), item->text());
                    showObjectDialog(this, m_baseAttributeView, "Geometry Object Details");
                }
            }
        }
    }
}

// ZGeometrySpreadsheet实现
ZGeometrySpreadsheet::ZGeometrySpreadsheet(QWidget* parent)
    : QWidget(parent)
    , m_views(new QStackedWidget(this))
    , m_clone_obj(nullptr)
{
    QLabel* pImgBlank = new QLabel("Current Object Is an image, please watch it in image panel");
    m_views->addWidget(pImgBlank);
    QLabel* pLblBlank = new QLabel("No object available, may be not apply or result is null");
    m_views->addWidget(pLblBlank);    //blank
    m_views->addWidget(new BaseAttributeView);
    m_views->addWidget(new SceneObjView);  // 添加SceneObjView组件
    m_views->addWidget(new ListObjView);   // 添加ListObjView组件
    m_views->addWidget(new MaterialObjView); // 添加MaterialObjView组件

    QVBoxLayout* pMainLayout = new QVBoxLayout;
    pMainLayout->setContentsMargins(1, 1, 1, 1);  // 去掉主布局边距
    pMainLayout->setSpacing(0);  // 设置间距为0
    pMainLayout->addWidget(m_views);
    setLayout(pMainLayout);

    connect(zenoApp->graphsManager(), &GraphsManager::fileClosed, this, [this]() {
        clearModel();
        });
}

void ZGeometrySpreadsheet::setGeometry(
        GraphModel* subgraph,
        QModelIndex nodeidx,
        std::unique_ptr<zeno::IObject>&& pObject
) {
    m_clone_obj = std::move(pObject);

    if (!m_clone_obj) {
        //unavailable page
        m_views->setCurrentIndex(1);
        return;
    }
    if (m_clone_obj->userData()->has("isImage")) {
        m_views->setCurrentIndex(0);  // 调整索引
        return;
    }

    if (auto geoObj = dynamic_cast<zeno::GeometryObject_Adapter*>(m_clone_obj.get())) {
        if (auto baseAttrView = qobject_cast<BaseAttributeView*>(m_views->widget(2))) {
            baseAttrView->setGeometryObject(subgraph, nodeidx, geoObj, nodeidx.data(QtRole::ROLE_NODE_NAME).toString());
        }
        m_views->setCurrentIndex(2);
    } else if (auto primObj = dynamic_cast<zeno::PrimitiveObject*>(m_clone_obj.get())) {
        if (auto convertedObj = create_GeometryObject(primObj)) {
            m_clone_obj = std::move(convertedObj);
            if (auto baseAttrView = qobject_cast<BaseAttributeView*>(m_views->widget(2))) {
                baseAttrView->setGeometryObject(subgraph, nodeidx, static_cast<zeno::GeometryObject_Adapter*>(m_clone_obj.get()), nodeidx.data(QtRole::ROLE_NODE_NAME).toString());
            }
            m_views->setCurrentIndex(2);
        }
    } else if (auto sceneObj = dynamic_cast<zeno::SceneObject*>(m_clone_obj.get())) {
        if (auto sceneObjView = qobject_cast<SceneObjView*>(m_views->widget(3))) {
            sceneObjView->setSceneObject(subgraph, nodeidx, sceneObj, nodeidx.data(QtRole::ROLE_NODE_NAME).toString());
        }
        m_views->setCurrentIndex(3);
    } else if (auto listObj = dynamic_cast<zeno::ListObject*>(m_clone_obj.get())) {
        if (auto listObjView = qobject_cast<ListObjView*>(m_views->widget(4))) {
            listObjView->setListObject(subgraph, nodeidx, listObj);
        }
        m_views->setCurrentIndex(4);
    } else if (auto materialObj = dynamic_cast<zeno::MaterialObject*>(m_clone_obj.get())) {
        if (auto materialObjView = qobject_cast<MaterialObjView*>(m_views->widget(5))) {
            materialObjView->setMaterialObject(subgraph, nodeidx, materialObj, nodeidx.data(QtRole::ROLE_NODE_NAME).toString());
        }
        m_views->setCurrentIndex(5);
    } else {
        m_views->setCurrentIndex(1);
    }
}

void ZGeometrySpreadsheet::clearModel()
{
    for (int i = 0; i < m_views->count(); ++i) {
        if (auto* view = qobject_cast<BaseAttributeView*>(m_views->widget(i))) {
            view->clearModel();
        } else if (auto* view = qobject_cast<SceneObjView*>(m_views->widget(i))) {
            view->clearModel();
        } else if (auto* view = qobject_cast<ListObjView*>(m_views->widget(i))) {
            view->clearModel();
        } else if (auto* view = qobject_cast<MaterialObjView*>(m_views->widget(i))) {
            view->clearModel();
        }
    }
}