! gfortran main.f90 -o main
program main
    implicit none

    integer, parameter :: dp = kind(1.d0)

    real(kind=dp), allocatable :: phi_r(:, :)
    complex(kind=dp), allocatable :: phi_k(:, :)

    real(kind=dp), allocatable :: xs(:), ys(:)
    real(kind=dp), allocatable :: kxs(:), kys(:)

    integer :: xn, yn, kxn, kyn
    real(kind=dp) :: x_length, y_length, kx_length, ky_length

    real(kind=dp) :: r(2), k(2)

    integer :: i, j, m, n

    ! read in phi_r matrix, including params
    ! (xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length)
    print *, "*** reading phi(r)"
    print *, "../data/phi_r_triple_SoD_basis-factor=1.8_lower=-1_small=2_upper=1.dat"

    open(unit=1, file="../data/phi_r_triple_SoD_basis-factor=1.8_lower=-1_small=2_upper=1.dat", status='old', action='read')
    read(1, *) xn
    read(1, *) yn
    read(1, *) x_length
    read(1, *) y_length
    read(1, *) kxn
    read(1, *) kyn
    read(1, *) kx_length
    read(1, *) ky_length

    print *, "xn=", xn, "yn=", yn
    print *, "kxn=", kxn, "kyn=", kyn

    allocate(xs(xn+1))
    allocate(ys(yn+1))
    allocate(kxs(kxn+1))
    allocate(kys(kyn+1))
    allocate(phi_r(xn, yn))

    do i=1, xn
        read(1, *) phi_r(i, :)
    end do

    close(1)

    allocate(phi_k(kxn, kyn))

    call make_axes(x_length, y_length, xn, yn, xs, ys)
    call make_axes(kx_length, ky_length, kxn, kyn, kxs, kys)

    print *, "*** DFT"
    ! preform FFT r->k
    do i=1, kxn
        do j=1, kyn
            k(1) = kxs(i)
            k(2) = kys(j)
            do m=1, xn
                do n=1, yn
                    r(1) = xs(m)
                    r(2) = ys(n)
                    phi_k(i, j) = phi_k(i, j) + phi_r(m, n) * EXP((0, 1) * DOT_PRODUCT(r, k))
                end do
            end do
        end do
        if (MOD(i, 20) == 0) then
            print *, "finished ", i
        end if
    end do

    print *, "*** writing phi(k)"
    print *, "phi_k_triple_SoD_basis-factor=1.8_lower=-1_small=2_upper=1.dat"

    ! write phi_k
    open(unit=1, file="phi_k_triple_SoD_basis-factor=1.8_lower=-1_small=2_upper=1.dat", status='replace', action='write')
    write(1, *) kxn
    write(1, *) kyn
    write(1, *) kx_length
    write(1, *) ky_length

    do i=1, kxn
        write(1, *) phi_k(i, :)
    end do

    close(1)

end program main

subroutine print_real_matrix(n, m, A)
    implicit none

    integer, parameter :: dp = kind(1.d0)

    integer, intent(in) :: n
    integer, intent(in) :: m
    real(kind=dp), intent(in) :: A(n,m)

    integer :: i

    do i=1, n
        !print *, A(i, :)
        write(*,*) A(i, :)
    end do

end subroutine print_real_matrix

subroutine make_axes(x_length, y_length, xn, yn, xs, ys)
    implicit none

    integer, parameter :: dp = kind(1.d0)

    real(kind=dp), intent(in) :: x_length, y_length
    integer, intent(in) :: xn, yn
    real(kind=dp), intent(out) :: xs(xn+1), ys(yn+1)

    real(kind=dp) :: dx, dy

    integer :: i

    dx = x_length / xn
    dy = y_length / yn

    xs(1) = -x_length / 2
    ys(1) = -y_length / 2

    do i=1, xn+1
        xs(i+1) = xs(i) + dx
    end do

    do i=1, yn+1
        ys(i+1) = ys(i) + dy
    end do

end subroutine make_axes