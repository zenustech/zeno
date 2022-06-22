#ifndef __VARIANT_PTR_H__
#define __VARIANT_PTR_H__

//TODO: a better file place.

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