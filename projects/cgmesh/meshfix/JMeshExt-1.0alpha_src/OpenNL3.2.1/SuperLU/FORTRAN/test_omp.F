! A simple OpenMP example to use SuperLU to solve multiple independent linear systems.
! Contributor: Ed D'Azevedo, Oak Ridge National Laboratory
!
       program tslu_omp
       implicit none
       integer, parameter :: maxn = 10*1000
       integer, parameter :: maxnz = 100*maxn
       integer, parameter :: nsys = 6  !! 64

       real*8 :: values(maxnz), b(maxn)
       integer :: rowind(maxnz), colptr(maxn)
!        integer :: Ai(maxnz, nsys), Aj(maxn, nsys)   ! Sherry added
       integer n, nnz, nrhs, ldb, info, iopt
       integer*8 :: factors, lufactors(nsys)
       real*8 :: A(maxnz, nsys)
       integer :: luinfo(nsys)
       real*8 :: brhs(maxn,nsys)
       integer :: i,j
       real*8 :: err, maxerr
       integer :: nthread
!$     integer, external :: omp_get_num_threads

!      --------------
!      read in matrix
!      --------------
       print*, 'before hbcode1'
       call hbcode1(n,n,nnz,values,rowind,colptr)
       print*, 'after hbcode1'

       nthread = 1
!$omp  parallel
!$omp  master
!$     nthread = omp_get_num_threads()
!$omp  end master
!$omp  end parallel
       write(*,*) 'nthreads = ',nthread
       write(*,*) 'nsys = ',nsys
       write(*,*) 'n, nnz ', n, nnz


!$omp  parallel do private(j)
       do j=1,nsys
          A(1:nnz,j) = values(1:nnz)
       enddo

       nrhs = 1
       ldb = n

!$omp  parallel do private(j)
       do j=1,nsys
          brhs(:,j) = j
       enddo


!      ---------------------
!      perform factorization
!      ---------------------
       iopt = 1

!$omp  parallel do private(j,values,b,info,factors)
       do j=1,nsys
!$omp  parallel workshare
         values(1:nnz) = A(1:nnz,j)
         b(1:n) = brhs(1:n,j)
!$omp  end parallel workshare
         info = 0

         call c_fortran_dgssv( iopt,n,nnz, nrhs, values, rowind, colptr,   &
     &          b, ldb, factors, info )

!$omp    parallel workshare
         A(1:nnz,j) = values(1:nnz)
         brhs(1:n,j) = b(1:n)
!$omp    end parallel workshare
         luinfo(j) = info
         lufactors(j) = factors
        enddo

       do j=1,nsys
         info = luinfo(j)
         if (info.ne.0) then
           write(*,9010) j, info
 9010      format(' factorization of j=',i7,' returns info= ',i7)
         endif
       enddo

!      ---------------------------------------
!      solve the system using existing factors
!      ---------------------------------------
       iopt = 2
!$omp  parallel do private(j,b,values,factors,info)
       do j=1,nsys
         factors = lufactors(j)
         values(1:nnz) = A(1:nnz,j)
         info = 0
         b(1:n) = brhs(1:n,j)
         call c_fortran_dgssv( iopt,n,nnz,nrhs,values,rowind,colptr,        &
     &            b,ldb,factors,info )
         lufactors(j) = factors
         luinfo(j) = info
         brhs(1:n,j) = b(1:n)
       enddo

!      ------------
!      simple check
!      ------------
       err = 0
       maxerr = 0

       do j=2,nsys
         do i=1,n
           err = abs(brhs(i,1)*j - brhs(i,j))
           maxerr = max(maxerr,err)
         enddo
       enddo
       write(*,*) 'max error = ', maxerr

!      -------------
!      free storage
!      -------------

       iopt = 3
!$omp  parallel do private(j)
       do j=1,nsys
          call c_fortran_dgssv(iopt,n,nnz,nrhs,A(:,j),rowind,colptr,     &
     &            brhs(:,j), ldb, lufactors(j), luinfo(j) )
       enddo
         
       stop
       end program
