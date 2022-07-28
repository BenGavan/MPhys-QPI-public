program main
    implicit none

    complex, allocatable :: x(:,:)
    integer :: n, m
    integer :: i

    open(unit=1, file="mat_complex.dat", status='old', action='read')
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
    complex, intent(in) :: A(n,m)

    integer :: i

    print *, "This is from reading a complex matrix!!"

    do i=1, n
        print *, A(i, :)
    end do

end subroutine print_matrix

!subroutine write_read_4_matrix()
!    implicit none
!
!    complex :: mat(4,4, 2,2)
!    integer :: i, j, k, l
!
!    do i=1, 4
!        do j=1, 4
!            do k=1, 2
!                do l=1, 2
!                    mat(i, j, k, l) = i*4 + 4*j + k*2 + l+2
!                end do
!            end do
!        end do
!    end do
!
!    open(unit=1, file="mat_4d.dat", status='replace', action='write')
!
!    write(1, *) 4
!    write(1, *) 4
!    write(1, *) 2
!    write(1, *) 2
!
!    do i=1, 4
!        write(1, *) mat(i,j, k, :)
!    end do
!
!    close(1)
!
!    complex, allocatable :: x(:,:)
!    integer :: n, m
!    integer :: i
!
!    open(unit=2, file="matrix_4d.dat", status="old", action)

!end subroutine write_read_4_matrix