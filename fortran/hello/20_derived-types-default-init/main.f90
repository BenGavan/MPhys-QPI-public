program main
    implicit none

    type :: t_pair
        integer :: i = 1
        real :: x = 2.4
    end type t_pair

    type(t_pair) :: pair
    type(t_pair) :: pairB

    print *, pair

    pairB = t_pair()

    print *, pairB

end program main