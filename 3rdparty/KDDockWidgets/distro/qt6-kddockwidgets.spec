Name:           qt6-kddockwidgets
Version:        1.4.0
Release:        1
Summary:        KDAB's Dock Widget Framework for Qt6
Source0:        %{name}-%{version}.tar.gz
Source1:        %{name}-%{version}.tar.gz.asc
Source2:        %{name}-rpmlintrc
URL:            https://github.com/KDAB/KDDockWidgets
Group:          System/Libraries
License:        GPL-2.0-only OR GPL-3.0-only
BuildRoot:      %{_tmppath}/%{name}-%{version}-build
Vendor:         Klaralvdalens Datakonsult AB (KDAB)
Packager:       Klaralvdalens Datakonsult AB (KDAB) <info@kdab.com>

BuildRequires: cmake
%if %{defined suse_version}
BuildRequires:  libqt6-qtbase-devel libqt6-qtx11extras-devel
%endif

%if %{defined fedora}
BuildRequires:  gcc-c++ qt6-qtbase-devel desktop-file-utils
%endif

%if %{defined rhel}
BuildRequires:  gcc-c++ qt6-qtbase-devel qt6-qtx11extras-devel desktop-file-utils
%endif

%description
KDDockWidgets is a Qt dock widget library written by KDAB, suitable for replacing
QDockWidget and implementing advanced functionalities missing in Qt, including:
 - Nesting dock widgets in a floating window and docking that group back to main window
  - Docking to any main window, not only to the parent main window
  - Docking to center of main window, or simply removing the concept of "central widget"
  - Main window supporting detachable tabs in center widget
  - Detaching arbitrary tabs from a tab bar into a dock area
  - Exposing inner helper widgets so the user can customize them or provide his own
    - Customize tab widgets
    - Customize title bars
    - Customize window frames
    - Custom widget separators
  ...and much more

Authors:
--------
      Klaralvdalens Datakonsult AB (KDAB) <info@kdab.com>

%define debug_package %{nil}
%global __debug_install_post %{nil}

%package devel
Summary:        Development files for %{name}
Group:          Development/Libraries/C and C++
Requires:       %{name} = %{version}

%description devel
This package contains header files and associated tools and libraries to
develop programs using kddockwidgets.

%prep
%autosetup

%build
cmake . -DCMAKE_INSTALL_PREFIX=/usr -DKDDockWidgets_QT6=True -DCMAKE_BUILD_TYPE=Release
%__make %{?_smp_mflags}

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%install
%make_install

%clean
%__rm -rf "%{buildroot}"

%files
%defattr(-,root,root)
%{_prefix}/share/doc/KDDockWidgets-qt6
%{_libdir}/libkddockwidgets-qt6.so.*

%files devel
%defattr(-,root,root,-)
%dir %{_includedir}/kddockwidgets-qt6
%{_includedir}/kddockwidgets-qt6/kddockwidgets/*
%dir %{_libdir}/cmake/KDDockWidgets-qt6
%{_libdir}/cmake/KDDockWidgets-qt6/*
%{_libdir}/libkddockwidgets-qt6.so
#%{_prefix}/mkspecs/modules/* ECMGeneratePriFile isn't ported to Qt6 yet

%changelog
* Fri Jul 16 2021 Allen Winter <allen.winter@kdab.com> 1.4.0
  1.4.0 final
* Mon Jun 07 2021 Allen Winter <allen.winter@kdab.com> 1.3.1
  1.3.1 final
* Mon Feb 08 2021 Allen Winter <allen.winter@kdab.com> 1.3.0
  1.3.0 final
* Thu Dec 17 2020 Allen Winter <allen.winter@kdab.com> 1.2.0
  1.2.0 final
* Fri Dec 11 2020 Allen Winter <allen.winter@kdab.com> 1.1.1
  1.1.1 final
* Mon Oct 26 2020 Allen Winter <allen.winter@kdab.com> 1.1.0
  1.1.0 final
* Wed Sep 02 2020 Allen Winter <allen.winter@kdab.com> 1.0.0
  1.0.0 final
* Thu Aug 06 2020 Allen Winter <allen.winter@kdab.com> 0.99.9
  1.0.0 release candidate
