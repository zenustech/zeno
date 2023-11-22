//
// Created by zh on 2023/3/23.
//

#ifndef ZENO_ZENOIMAGEPANEL_H
#define ZENO_ZENOIMAGEPANEL_H

#include <QtWidgets>
#include <zenoui/comctrl/zcombobox.h>

class ZenoImageView: public QGraphicsView {
    Q_OBJECT
public:
    float ziv_wheelZoomFactor = 1.25;
    QGraphicsPixmapItem *_image = nullptr;
    QGraphicsScene *scene = nullptr;
    bool fitMode = true;
    explicit ZenoImageView(QWidget *parent) : QGraphicsView(parent) {
        scene = new QGraphicsScene;
        this->setScene(scene);
        setMouseTracking(true);

        setBackgroundBrush(QColor(37, 37, 37));
    }

    bool hasImage() const {
        return _image != nullptr;
    }

    void clearImage() {
        if (hasImage()) {
            scene->removeItem(_image);
            _image = nullptr;
        }
    }

    void setImage(const QImage &image) {
        QPixmap pm = QPixmap::fromImage(image);
        if (hasImage()) {
            _image->setPixmap(pm);
        }
        else {
            _image = this->scene->addPixmap(pm);
        }
        setSceneRect(QRectF(pm.rect()));  // Set scene size to image size.
        updateImageView();
    }

    void updateImageView() {
        if (!hasImage()) {
            return;
        }
        if (fitMode) {
            fitInView(sceneRect(), Qt::AspectRatioMode::KeepAspectRatio);
        }
    }
signals:
    void pixelChanged(float, float);
protected:
    void resizeEvent(QResizeEvent *event) override {
        updateImageView();
    }
    void wheelEvent(QWheelEvent* event) override {
        fitMode = false;
        qreal zoomFactor = 1;
        if (event->angleDelta().y() > 0)
            zoomFactor = ziv_wheelZoomFactor;
        else if (event->angleDelta().y() < 0)
            zoomFactor = 1 / ziv_wheelZoomFactor;
        scale(zoomFactor, zoomFactor);
    }
    void mousePressEvent(QMouseEvent* event) override {
        fitMode = false;
        setDragMode(QGraphicsView::ScrollHandDrag);
        QGraphicsView::mousePressEvent(event);
    }
    void mouseReleaseEvent(QMouseEvent* event) override {
        QGraphicsView::mouseReleaseEvent(event);
        setDragMode(QGraphicsView::NoDrag);
    }
    void mouseMoveEvent(QMouseEvent* event) override {
        if(auto *item = qgraphicsitem_cast<QGraphicsPixmapItem *>(itemAt(event->pos()))){
            QPointF p = item->mapFromScene(mapToScene(event->pos()));
            emit(pixelChanged(p.x(), p.y()));
        }
        QGraphicsView::mouseMoveEvent(event);
    }
};

class ZenoImagePanel : public QWidget {
    Q_OBJECT

    QLabel* pStatusBar = new QLabel();
    QLabel* pPrimName = new QLabel();
    QCheckBox *pGamma = new QCheckBox("Gamma");
    ZComboBox *pMode = new ZComboBox();
    QPushButton *pFit = new QPushButton("Fit");
    ZenoImageView *image_view = nullptr;

public:
    ZenoImagePanel(QWidget* parent = nullptr);
    void clear();
    void setPrim(std::string primid);
};


#endif //ZENO_ZENOIMAGEPANEL_H
