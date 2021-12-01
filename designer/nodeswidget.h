#ifndef __NODESWIDGET_H__
#define __NODESWIDGET_H__

class NodesView;

class NodesWidget : public QWidget
{
    Q_OBJECT
public:
    NodesWidget(QWidget *parent = nullptr);
    NodesWidget(const QString& filePath, QWidget* parent = nullptr);
    QStandardItemModel* model() const;
    QItemSelectionModel *selectionModel() const;
    SnapWay getSnapWay() const { return m_snap; }
    qreal factor() const;
    bool showBorder() const { return m_bShowBdr; }
    void markDirty(bool dirty);
    QString fileName() const;

signals:
    void tabDirtyChanged(bool);

public slots:
    void setFactor(const qreal &factor);
    void save();
    void resetPreset();

private:
    void init(const QString &filename);

    NodesView *m_pView;
    SnapWay m_snap;
    qreal m_factor;
    QString m_fileName;
    QString m_filePath;
    bool m_bShowBdr;
    bool m_dirty;
};


#endif