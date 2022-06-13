# Maintainer: Yubin Peng <1931127624@qq.com>
pkgname=zeno
pkgver=r11.e6cf2fd
pkgrel=1
pkgdesc="Open-source node system framework for simulation and others"
arch=('x86_64')
url='https://github.com/zenustech/zeno'
license=('MPL2')
depends=("qt5-base" "qt5-svg" "tbb" "openvdb" "eigen" "cgal" "lapack" "openblas" "alembic" "hdf5")
makedepends=("git" "cmake" "ninja")
optdepends=()
source=("${pkgname}::git+${url}.git")
noextract=()
md5sums=('SKIP')

pkgver() {
    printf "r%s.%s" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

prepare() {
    cd "${pkgname}"
    git submodule update --init projects/cgmesh/libigl
    cp misc/ci/CMakePresets.json ./
}

build() {
    cd "${pkgname}"
    cmake --preset default -G Ninja -DCMAKE_INSTALL_PREFIX="${pkgdir}" -DZENO_SYSTEM_ALEMBIC:BOOL=ON -DZENO_SYSTEM_OPENVDB:BOOL=ON
    cmake --build --preset default --parallel
}

package() {
    cd "${pkgname}"
    cmake --build --preset default --target install
}
