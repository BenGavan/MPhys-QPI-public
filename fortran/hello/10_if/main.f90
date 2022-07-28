program main
    implicit none

    real :: angle

    angle = 189.9

    if (angle < 90.0) then
        print *, 'Angle is acute'
    else if (angle < 180) then
        print *, "Angle is obtuse"
    else
        print *, 'Angle is reflex'
    end if


end program main