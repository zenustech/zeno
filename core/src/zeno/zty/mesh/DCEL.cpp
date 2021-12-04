#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>


ZENO_NAMESPACE_BEGIN
namespace zty {


DCEL::DCEL() noexcept = default;
DCEL::DCEL(DCEL &&that) noexcept = default;
DCEL &DCEL::operator=(DCEL &&that) noexcept = default;
DCEL::DCEL(DCEL const &that) = default;
DCEL &DCEL::operator=(DCEL const &that) = default;


}
ZENO_NAMESPACE_END
