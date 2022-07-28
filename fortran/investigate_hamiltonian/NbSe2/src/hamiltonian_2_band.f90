module m_hamiltonian_2_band
    implicit none
    public get_hi

contains

    subroutine get_hi(s)
        implicit none
        character(*), intent(out) :: s

        s = "hello"

    end subroutine get_hi

end module m_hamiltonian_2_band