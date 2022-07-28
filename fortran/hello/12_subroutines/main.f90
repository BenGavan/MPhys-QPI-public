module m_something
    implicit none
    private
    public calc_something


contains
    subroutine calc_something(filepath, n, m, matrix)
        implicit none
        character(*), intent(in) :: filepath
        integer, intent(out) :: n
        integer, intent(out) :: m
        real, allocatable, intent(out) :: matrix(:,:)

        n = 3
        m = 4

        allocate(matrix(n, m))

        matrix(:, :) = 1

    end subroutine calc_something

end module m_something

program main
    use m_something
    implicit none

!    real :: mat(10, 20)
    real, allocatable :: something_matrix(:,:)

    integer :: n, m

!    mat(:,:) = 0.0

!    call print_matrix(10, 20, mat)

    call calc_something("a string", n, m, something_matrix)
    call print_matrix(n, m, something_matrix)

end program main

subroutine print_matrix(n,m,A)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: m
    real, intent(in) :: A(n,m)

    integer :: i

    do i=1, n
        print *, A(i, 1:m)
    end do

end subroutine print_matrix