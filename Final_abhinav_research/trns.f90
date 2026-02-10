      subroutine trns (np1,np2,np,p1,p2,p3,xp,ap)

!------------------------------------------------------------------------
!
!     trns calculates the vectors xp und ap with transformed 
!     gauss-legendre-points and weights
!
!     np1 points are mapped with the hyperbolic transformation
!
!     x --> (1.+x) / (1./p1-(1./p1-2./p2)*x)
!
!     to the  interval (0.;p2) , where
!     np1/2 points are in (0.;p1) and
!     np1/2 points are in (p1;p2) 
!
!     np2 points are mapped with the linear transformation
!
!     x --> (p3+p2)/2. + (p3-p2)/2.*x
!
!     to the interval (p2;p3) 
!
!     np = np1 + np2
!


      implicit none

      integer :: i, np1, np2, np
      double precision  :: x, xx, a, delph, p1, p2, p3

      double precision, dimension(np1) :: xp1, ap1
      double precision, dimension(np2) :: xp2, ap2
      double precision, dimension(np)  :: xp, ap

      call gauleg(-1.d0,1.d0,xp1,ap1,np1)

      do i = 1,np1
       x      = xp1(i)
       a      = ap1(i)
       xx     = 1.d0/p1-(1.d0/p1-2.d0/p2)*x
       xp1(i) = (1.d0+x) / xx
       ap1(i) = (2.d0/p1-2.d0/p2)*a / xx**2
      end do

      if (np2 .ne. 0) then
       call gauleg (-1.d0,1.d0,xp2,ap2,np2)
       do i = 1,np2
          x      = xp2(i)
          a      = ap2(i)
          delph  = (p3-p2)/2.d0
          xp2(i) = (p3+p2)/2.d0 + delph * x
          ap2(i) = delph * a
       end do
      endif

      do i = 1,np1
       xp(i) = xp1(i)
       ap(i) = ap1(i)
      end do

      if (np2 .ne. 0) then
       do i = 1,np2
          xp(i+np1) = xp2(i)
          ap(i+np1) = ap2(i)
       end do
      endif


      end subroutine trns


