module m_consts
    implicit none
    private
    public t_params

    type :: t_params
        real :: omega = -3.925  ! eV
        real :: kxn = 600
        real :: kyn = 600
    end type t_params
end module m_consts

program main
    use m_consts
    implicit none

    type(t_params) :: params

    print *, params

end program main