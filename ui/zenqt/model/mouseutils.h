#ifndef MOUSEUTILS_H
#define MOUSEUTILS_H

#include <QObject>
#include <QCursor>
#include <QPoint>

class MouseUtils : public QObject {
    Q_OBJECT
public:
    explicit MouseUtils(QObject* parent = nullptr) : QObject(parent) {}

    Q_INVOKABLE QPoint getGlobalMousePosition() const {
        return QCursor::pos();
    }
};

#endif // MOUSEUTILS_H
