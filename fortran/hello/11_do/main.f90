program main
    implicit none

    integer :: i

    do i = 1, 100
        if (mod(i, 2) == 0) then
            cycle
        end if
        if (i > 15) then
            exit
        end if
        print *, i
    end do

    print *, 'end'

end program main