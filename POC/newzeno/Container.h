
namespace zeno::v2::container {

    struct _AnyBase {
        virtual ~_AnyBase() = default;

        virtual () = 0;
    };

    template <class T>
    struct _AnyImpl : _AnyBase {
        T value;
    };

}
