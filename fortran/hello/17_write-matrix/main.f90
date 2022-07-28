! gfortran main.f90 -o main
program main
    implicit none

    complex :: mat(3, 5)

    integer :: i
    integer :: j

    do i=1, 3
        do j=1, 5
            mat(i,j) = (i*3)+j
        end do
    end do

    call print_matrix(3, 5, mat)

    open(unit=1, file="mat_complex.dat", status='replace', action='write')

    write(1, *) 3
    write(1, *) 5

    do i=1, 3
        write(1, *) mat(i,:)
    end do

    close(1)

end program main

subroutine print_matrix(n, m, A)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: m
    complex, intent(in) :: A(n,m)

    integer :: i

    do i=1, n
        print *, A(i,:)
    end do
end subroutine print_matrix