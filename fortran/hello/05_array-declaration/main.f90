program main
    implicit none

    ! Array Declaration !
    ! 1D int array
    integer, dimension(10) :: array1

    ! equivalent
    integer :: array2(10)

    ! 2D real array
    real, dimension(10, 10) :: array2D

    ! Custom lower and upper index bounds
    real :: array4(0:9)
    real :: array5(-5:5)

    print *, array1
    array1(1) = 69
    print *, array1

    print *, "---------"
    print *, array2D
    array2D(1,1) = 420
    print *, "---------"
    print *, array2D
end program main