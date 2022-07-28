! Based on: https://stackoverflow.com/questions/8828377/reading-data-from-txt-file-in-fortran
program main
    implicit none

    real, allocatable :: x(:,:)
    integer :: n, m
    integer :: i

    open(unit=1, file="mat.dat", status='old', action='read')
    read(1, *) n
    read(1, *) m

    allocate(x(n,m))

    do i=1, n
        read(1, *) x(i,:)
    end do

    close(1)

    call print_matrix(n, m, X)

end program main

subroutine print_matrix(n, m, A)
    implicit none

    integer, intent(in) :: n
    integer, intent(in) :: m
    real, intent(in) :: A(n,m)

    integer :: i

    do i=1, n
        print *, A(i, :)
    end do

end subroutine print_matrix