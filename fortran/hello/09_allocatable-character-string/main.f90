program main
    implicit none

    character(:), allocatable :: first_name
    character(:), allocatable :: last_name

    ! Explicit allocation
    allocate(character(len=3) :: first_name)
    first_name = "Ben"

    ! Allocation on assignment
    last_name = "Gavan"

    print *, first_name//last_name

end program main