! gfortran main.f90 -o main

module m_consts
    implicit none
    private
    public t_params

    integer, parameter ::                              &
            sp = kind(1.0),                                &
            dp = selected_real_kind(2*precision(1.0_sp)),  &
            qp = selected_real_kind(2*precision(1.0_dp))

    real(dp) :: PI=4.D0*DATAN(1.D0)

    type :: t_params
        real(dp) :: a_0 = 3.44   ! \AA (Angstrom)
        real :: omega = -3.925  ! eV
        integer :: kxn = 600
        integer :: kyn = 600
        real :: ky_length = 1
    end type t_params
end module m_consts

module m_nbse2_hamiltonian
    implicit none
    private
    public t_ham_r, get_2_band_hamiltonian !, make_18_from_2

    type :: t_ham_r
        complex, allocatable :: ham_r(:)
        real, allocatable :: weights(:)
        real, allocatable :: hopping_vectors(:,:)
    end type t_ham_r

contains
    subroutine get_2_band_hamiltonian(filepath, ham_r, weights, hopping_vectors)
        implicit none
        character(*), intent(in) :: filepath
        complex, allocatable, intent(out) :: ham_r(:,:,:)
        real, allocatable, intent(out) :: weights(:)
        real, allocatable, intent(out) :: hopping_vectors(:,:)
        integer :: num_of_bands
        integer :: num_of_hopping_vectors
        integer :: i, j, k, n, m
        real :: x, y, z, h_real, h_imag

        open(unit=1, file=filepath, action='read')

        read(1, *)
        read(1, *) num_of_bands
        read(1, *) num_of_hopping_vectors

        allocate(weights(num_of_hopping_vectors))
        allocate(hopping_vectors(num_of_hopping_vectors, 2))
        allocate(ham_r(num_of_hopping_vectors, num_of_bands, num_of_bands))

        print *, num_of_bands
        print *, num_of_hopping_vectors

        read(1, *) weights

        do i=1, num_of_hopping_vectors
            do j=1, num_of_bands * num_of_bands
                read(1, *) x, y, z, n, m, h_real, h_imag
!                print *, x, y, z, n, m, h_real, h_imag
                ham_r(i, n, m) = complex(h_real, h_imag)
                if (j==1) then
                    hopping_vectors(i, 1) = x
                    hopping_vectors(i, 2) = y
!                    print *, hopping_vectors(i, :)
                end if
            end do
        end do

    end subroutine get_2_band_hamiltonian

!    subroutine make_18_from_2(ham_2bands_r, weights, hopping_vectors, ham_18bands_r)
!        implicit none
!        complex, intent(in) :: ham_2bands_r(:,:,:)
!        real, allocatable, intent(in) :: weights(:)
!        real, allocatable, intent(in) :: hopping_vectors(:,:)
!        complex, allocatable, intent(out) :: ham_18bands_r(:,:,:)
!
!        integer :: min_1, max_1, min_2, max_2
!        integer :: i
!
!        integer :: num_hopping_vectors = SIZE(weights)
!
!        ! get min/max vectors to determine lookup matrix size
!        min_1 = hopping_vectors(1,1)
!        min_2 = hopping_vectors(1,2)
!        max_1 = hopping_vectors(1,1)
!        max_2 = hopping_vectors(1,2)
!        do i=2, num_hopping_vectors
!            if (hopping_vectors(i, 1) > max_1) then
!                max_1 = hopping_vectors(i, 1)
!            end if
!            if (hopping_vectors(i, 2) > max_2) then
!                max_2 = hopping_vectors(i, 2)
!            end if
!
!            if (hopping_vectors(i, 1) < min_1) then
!                min_1 = hopping_vectors(i, 1)
!            end if
!        end do
!
!    end subroutine make_18_from_2

end module m_nbse2_hamiltonian

program main
    use m_consts
    use m_nbse2_hamiltonian
    use m_hamiltonian_2_band
    implicit none

    type(t_params) :: params
    complex, allocatable :: ham_r(:,:,:)
    real, allocatable :: weights(:)
    real, allocatable :: hopping_vectors(:,:)
    integer :: i
    character(10) :: s

    integer, parameter ::                              &
            sp = kind(1.0),                                &
            dp = selected_real_kind(2*precision(1.0_sp)),  &
            qp = selected_real_kind(2*precision(1.0_dp))

    real(dp) :: PI = 4.D0*DATAN(1.D0)

    real(dp) :: kx_length
    real(dp) :: ky_length

    kx_length = 4.D0 * Params%a_0 / PI
    ky_length = 4.D0 * Params%a_0 / PI

    print *, "-------------------------------"
    print *, "Investigate Hamiltonian - NbSe2"
    print *, "-------------------------------"

    print *, "kxn: ", params%kxn
    print *, "kyn: ", params%kyn

    print *, "kx_length:", kx_length
    print *, "ky_length:", ky_length


    call get_2_band_hamiltonian("../Hamiltonian.dat", ham_r, weights, hopping_vectors)

    print *, "SIZEOF(weights(1)) = ", SIZEOF(weights(1))
    print *, "weights(1) = ", weights(1)
    print *, "SIZEOF(weights) = ", SIZEOF(weights)
    print *, "SIZEOF(weights) / SIZEOF(weights(1)) = ", SIZEOF(weights) / SIZEOF(weights(1))
!    do i=1,

    call get_hi(s)

    print *, s
end program main