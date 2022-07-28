program main
    implicit none

    real :: v(9)
    real :: vector_norm
    real :: help_me
!    real :: this_is_contained

    real :: value_for_test_in_func

    value_for_test_in_func = 10.1

    v(:) = 9

    print *, v**2

    print *, 'Vector norm = ', vector_norm(9, v)

    print *, 'Vector norm = ', vector_norm(-9, v)

    print *, 'What = ', help_me(10)
    print *, 'contained = ', this_is_contained(2.0)

    contains

        function this_is_contained(x) result(f)
            implicit none
            real, intent(in) :: x
            real :: f

            f = value_for_test_in_func

!            f = x*1.5

        end function this_is_contained

end program main

function vector_norm(n, vec) result(norm)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: vec(n)
    real :: norm




    norm = sqrt(sum(vec**2))

end function vector_norm


function help_me(n) result(x)
    implicit none
    integer, intent(in) :: n
    real :: x

    x = sqrt(real(n))

end function help_me