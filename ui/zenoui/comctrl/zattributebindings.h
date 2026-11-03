#pragma once
#include <QWidget>
#include <QTableWidget>
#include <QStyledItemDelegate>
#include <QHeaderView>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QSet>
#include <QMap>
#include <QSignalBlocker>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QVBoxLayout>
#include <QLabel>
#include <QGraphicsProxyWidget>
#include <functional>

// Used by both native control factories: edits on the graph and in properties
// share the normal model callback, including undo/redo and graph serialization.
class ZAttributeBindingsEditor : public QWidget {
    // The application's item style adds text padding. Embedded controls occupy
    // the cell itself, so do not apply that text offset to their geometry.
    class CellDelegate : public QStyledItemDelegate {
    public:
        using QStyledItemDelegate::QStyledItemDelegate;
        void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option,
                                  const QModelIndex &) const override {
            editor->setGeometry(option.rect);
        }
    };
public:
    std::function<void(const QString&)> valueEdited;
    explicit ZAttributeBindingsEditor(const QString &value, QWidget *parent=nullptr, qreal scale=1)
        : QWidget(parent), m_scale(scale) {
        setObjectName("alternativeAttributeBindings");
        setStyleSheet("#alternativeAttributeBindings { background: transparent; }");
        setAutoFillBackground(false);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        setMinimumWidth(px(360));
        setFixedHeight(px(180));
        auto layout=new QVBoxLayout(this);layout->setContentsMargins(0,0,0,0);layout->setSpacing(px(6));
        m_table=new QTableWidget(0,4,this);m_table->setObjectName("attributeBindings");
        m_table->setProperty("cssClass", "proppanel");
        m_table->setItemDelegate(new CellDelegate(m_table));
        QFont tableFont=font(); tableFont.setPointSize(9); m_table->setFont(tableFont);
        m_table->setHorizontalHeaderLabels({tr("Source attribute"),tr("Domain"),tr("Shader attr"),tr("Alter")});
        const QStringList tips={
            tr("Name of the attribute on the incoming prim, for example detailUV. Corner sources use polygon loops or triangle/quad corner attributes."),
            tr("Where source values are stored on the prim. Point values remap to the primary-UV compacted vertices; face is per face; corner is per face corner."),
            tr("ShaderInputAttr attr selection: uv, nrm, clr or tang. Normal/tangent sources are object-space directions; their shader outputs are world-space. Colors are linear RGB."),
            tr("ShaderInputAttr alter selection, from 1 to 8. Zero is reserved for the default buffer.")};
        for(int column=0;column<tips.size();++column)
            m_table->horizontalHeaderItem(column)->setToolTip(tips[column]);
        m_table->horizontalHeader()->setSectionResizeMode(0,QHeaderView::Stretch);
        for(int column=1;column<4;++column)
            m_table->horizontalHeader()->setSectionResizeMode(column,QHeaderView::Fixed);
        m_table->setColumnWidth(1,px(80));
        m_table->setColumnWidth(2,px(88));
        m_table->setColumnWidth(3,px(64));
        m_table->horizontalHeader()->setMinimumHeight(px(28));
        m_table->verticalHeader()->setVisible(false);
        m_table->verticalHeader()->setDefaultSectionSize(px(30));
        m_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_table->setSelectionBehavior(QAbstractItemView::SelectRows);layout->addWidget(m_table);
        m_error=new QLabel(this);
        m_error->setObjectName("attributeBindingsError");
        m_error->setFont(tableFont);
        m_error->setTextFormat(Qt::PlainText);
        m_error->setWordWrap(true);
        m_error->setStyleSheet("color: #F28B82;");
        m_error->hide();layout->addWidget(m_error);
        auto actions=new QHBoxLayout;auto add=m_add=new QPushButton(tr("Add binding"),this);
        auto remove=new QPushButton(tr("Remove selected"),this);
        for(auto button:{add,remove}) {
            button->setProperty("cssClass","proppanel");
            button->setMinimumSize(px(112),px(28));
        }
        actions->setSpacing(px(6));
        actions->addWidget(add);actions->addWidget(remove);actions->addStretch();layout->addLayout(actions);
        connect(add,&QPushButton::clicked,this,[this]{addRow({});commit();});
        connect(remove,&QPushButton::clicked,this,[this]{
            const auto selected=m_table->selectionModel()->selectedRows();QSet<int> rows;
            for(const auto &index:selected)rows.insert(index.row());
            m_loading=true;for(int row=m_table->rowCount()-1;row>=0;--row)if(rows.contains(row))m_table->removeRow(row);
            m_loading=false;commit();
        });
        connect(m_table,&QTableWidget::itemChanged,this,[this]{commit();});setValue(value);
    }
    QSize sizeHint() const override {return {px(520),px(180)};}
    QString value() const {return m_value;}
    void setValue(const QString &value) {
        if(value==m_value && m_initialized)return;
        QJsonParseError error;auto document=QJsonDocument::fromJson(value.toUtf8(),&error);
        if(error.error!=QJsonParseError::NoError || !document.isArray()) {
            m_error->setText(tr("Invalid saved bindings"));m_error->show();return;
        }
        m_loading=true;m_table->setRowCount(0);
        for(const auto &record:document.array())addRow(record.toObject());
        m_loading=false;m_value=value;m_initialized=true;validateBindings();
    }
private:
    static constexpr int MaxAlter=8;
    static const QStringList &attributes() {
        static const QStringList names={"uv","nrm","clr","tang"};return names;
    }
    int px(int value) const {return qRound(value*m_scale);}
    void addRow(const QJsonObject &record) {
        const bool previous=m_loading;m_loading=true;
        QMap<QString,QSet<int>> occupied;
        for(int i=0;i<m_table->rowCount();++i)
            occupied[qobject_cast<QComboBox*>(m_table->cellWidget(i,2))->currentText()]
                .insert(qobject_cast<QSpinBox*>(m_table->cellWidget(i,3))->value());
        QString preferred="uv";
        if(m_table->currentRow()>=0)
            preferred=qobject_cast<QComboBox*>(m_table->cellWidget(m_table->currentRow(),2))->currentText();
        auto firstFree=[&](const QString &attr){int slot=1;while(occupied[attr].contains(slot))++slot;return slot;};
        auto semantic=record.value("attr").toString(preferred);
        if(!record.contains("attr") && (!attributes().contains(semantic) || firstFree(semantic)>MaxAlter))
            for(const auto &candidate:attributes())if(firstFree(candidate)<=MaxAlter){semantic=candidate;break;}
        const int next=firstFree(semantic);
        const int row=m_table->rowCount();m_table->insertRow(row);
        auto source=new QTableWidgetItem(record.value("source").toString(semantic));
        source->setToolTip(m_table->horizontalHeaderItem(0)->toolTip());
        m_table->setItem(row,0,source);
        auto domain=new QComboBox(m_table);domain->addItems({"point","face","corner"});domain->setCurrentText(record.value("domain").toString("point"));m_table->setCellWidget(row,1,domain);
        auto attr=new QComboBox(m_table);attr->addItems(attributes());if(attr->findText(semantic)<0)attr->addItem(semantic);
        attr->setCurrentText(semantic);m_table->setCellWidget(row,2,attr);
        auto alter=new QSpinBox(m_table);
        const auto savedAlter=record.value("alter").toInt(next);
        // Keep invalid legacy values visible until corrected, rather than
        // silently rebinding an old Alter ID to slot 8 when a graph loads.
        alter->setRange(qMin(1,savedAlter),qMax(MaxAlter,savedAlter));
        alter->setValue(savedAlter);m_table->setCellWidget(row,3,alter);
        domain->setToolTip(m_table->horizontalHeaderItem(1)->toolTip());
        attr->setToolTip(m_table->horizontalHeaderItem(2)->toolTip());
        alter->setToolTip(m_table->horizontalHeaderItem(3)->toolTip());
        connect(domain,&QComboBox::currentTextChanged,this,[this]{commit();});connect(attr,&QComboBox::currentTextChanged,this,[this]{commit();});
        connect(alter,qOverload<int>(&QSpinBox::valueChanged),this,[this,alter](int value){
            if(value>=1 && value<=MaxAlter) {
                const QSignalBlocker blocker(alter);alter->setRange(1,MaxAlter);
            }
            if(!m_loading)validateBindings();
        });
        connect(alter,&QSpinBox::editingFinished,this,[this]{commit();});m_loading=previous;
    }
    void validateBindings() {
        QMap<QPair<QString,int>,QVector<int>> bindings;
        QMap<int,QString> rowErrors;
        QSet<QPair<QString,int>> occupied;
        QString error;
        for(int row=0;row<m_table->rowCount();++row) {
            const auto attr=qobject_cast<QComboBox*>(m_table->cellWidget(row,2))->currentText();
            const auto alter=qobject_cast<QSpinBox*>(m_table->cellWidget(row,3))->value();
            bindings[qMakePair(attr,alter)].append(row);
            if(alter<1 || alter>MaxAlter) {
                const auto message=tr("Row %1: %2/%3 is outside the supported Alter range 1–8.")
                    .arg(row+1).arg(attr).arg(alter);
                rowErrors.insert(row,message);if(error.isEmpty())error=message;
            } else if(!attributes().contains(attr)) {
                const auto message=tr("Row %1: unknown Shader attr %2.").arg(row+1).arg(attr);
                rowErrors.insert(row,message);if(error.isEmpty())error=message;
            } else occupied.insert(qMakePair(attr,alter));
        }
        const bool hasFreeSlot=occupied.size()<attributes().size()*MaxAlter;
        m_add->setEnabled(hasFreeSlot);
        m_add->setToolTip(hasFreeSlot ? tr("Add the first free Alter slot (1–8) for the selected Shader attr, or the next attribute with a free slot.") :
            tr("All 32 alternative slots are occupied. Remove a binding before adding another."));
        for(auto it=bindings.cbegin();it!=bindings.cend();++it) {
            if(it.value().size()<2)continue;
            const auto message=tr("Duplicate %1/%2 (rows %3, %4). Alter must be unique per Shader attr.")
                .arg(it.key().first).arg(it.key().second)
                .arg(it.value()[0]+1).arg(it.value()[1]+1);
            for(int row:it.value())rowErrors.insert(row,message);
            if(error.isEmpty())error=message;
        }
        // Background/tool-tip updates also emit itemChanged. They are display
        // state, so keep them out of the graph's edit and undo callbacks.
        const QSignalBlocker blocker(m_table);
        for(int row=0;row<m_table->rowCount();++row) {
            const bool conflict=rowErrors.contains(row);
            m_table->item(row,0)->setBackground(conflict ? QBrush(QColor(90,36,36)) : QBrush());
            for(int column:{2,3}) {
                auto control=m_table->cellWidget(row,column);
                if(control->property("bindingConflict").toBool()!=conflict) {
                    control->setProperty("bindingConflict",conflict);
                    control->setStyleSheet(conflict ? "border: 1px solid #F28B82;" : "");
                }
                control->setToolTip(conflict ? rowErrors.value(row) : m_table->horizontalHeaderItem(column)->toolTip());
            }
        }
        m_error->setText(error);
        m_error->setVisible(!error.isEmpty());
    }
    void commit() {
        if(m_loading)return;
        validateBindings();
        QJsonArray rows;
        for(int i=0;i<m_table->rowCount();++i)rows.append(QJsonObject{
            {"source",m_table->item(i,0)->text()},
            {"domain",qobject_cast<QComboBox*>(m_table->cellWidget(i,1))->currentText()},
            {"attr",qobject_cast<QComboBox*>(m_table->cellWidget(i,2))->currentText()},
            {"alter",qobject_cast<QSpinBox*>(m_table->cellWidget(i,3))->value()}});
        const auto value=QString::fromUtf8(QJsonDocument(rows).toJson(QJsonDocument::Compact));
        if(value==m_value)return;m_value=value;if(valueEdited)valueEdited(value);
    }
    QTableWidget *m_table{};QLabel *m_error{};QPushButton *m_add{};QString m_value;bool m_loading{},m_initialized{};qreal m_scale;
};

template<class Proxy=QGraphicsProxyWidget>
inline Proxy *createAttributeBindingsItem(const QString &value,std::function<void(const QString&)> edit,qreal scale=1) {
    auto proxy=new Proxy;auto editor=new ZAttributeBindingsEditor(value,nullptr,scale);
    // Align graph contents with the socket text (10px socket + 8px spacing).
    editor->layout()->setContentsMargins(qRound(18*scale),0,0,0);
    editor->valueEdited=std::move(edit);proxy->setWidget(editor);editor->setAutoFillBackground(false);proxy->resize(editor->sizeHint());return proxy;
}
