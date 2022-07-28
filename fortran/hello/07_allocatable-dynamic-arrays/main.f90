program main
    implicit none

    ! NOTE: Allocatable local arrays are deallocted automatically when they go out of scope

    integer, allocatable :: array1(:)
    integer, allocatable :: array2(:,:)

    print *, array1
    print *, array2

    allocate(array1(10))
    allocate(array2(10,10))

    print *, array1
    print *, array2

    deallocate(array1)
    deallocate(array2)

    print *, array1
    print *, array2

end program main