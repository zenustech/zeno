#ifndef __VARIANT_PTR_H__
#define __VARIANT_PTR_H__

template<class T>
class QVariantPtr
{
public:
    static T* asPtr(QVariant v)
    {
        return (T*)v.value<void*>();
    }
    static QVariant asVariant(T* ptr)
    {
        return QVariant::fromValue((void *)ptr);    
    }
};

#endif