program main
    implicit none

    type :: t_pair
        integer :: i
        real :: x
    end type t_pair

    type(t_pair) :: pair

    pair%i = 1
    pair%x = 2

    print *, "pair.i = ", pair%i, ", pair.x = ", pair%x

    pair = t_pair(3, 3.4)

    print *, "pair.i = ", pair%i, ", pair.x = ", pair%x

    pair = t_pair(i=5, x=5.6)

    print *, "pair.i = ", pair%i, ", pair.x = ", pair%x

    pair = t_pair(i=7, x=7.89)

    print *, "pair.i = ", pair%i, ", pair.x = ", pair%x

end program main

!type :: t_example
!    ! 1st case: simple built-in type with access attribute and [init]
!    integer, private :: i = 0
!    ! private hides it from use outside of the t_example's scope.
!    ! The default initialization [=0] is the [init] part.
!
!    ! 2nd case: protected
!    integer, protected :: i
!    ! In contrary to private, protected allows access to i assigned value outside of t_example
!    ! but is not definable, i.e. a value may be assigned to i only within t_example.
!
!    ! 3rd case: dynamic 1-D array
!    real, allocatable, dimension(:) :: x
!    ! the same as
!    real, allocatable :: x(:)
!    ! This parentheses' usage implies dimension(:) and is one of the possible [attr-dependent-spec].
!end type t_example



!type :: t_example
!    integer, private :: i = 1
!
!    integer, public :: j = 2
!end type
