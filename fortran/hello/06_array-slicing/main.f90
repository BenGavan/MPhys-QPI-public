program main
    implicit none

    integer :: i
    integer :: array1(10)  ! static array (size fixed at comilation)
    integer :: array1D(10)
    integer :: array2D(10, 10)

    print *, array1
    array1 = [1,2,3,4,5,6,7,8,9,10]
    print *, array1

    print *, array1D
    array1D = [(i, i = 1, 10)]
    print *, array1D

    array1D(:) = 0  ! set all elements to 0
    print *, array1D

    array1D(1:5) = 1  ! set first 5 elements to 1
    print *, array1D

    array1D(6:) = 2  ! set all elements after 5 to 2
    print *, array1D

    print *, array1D(1:10:2)  ! elements at odd indices (every 2 indices)
    print *, array1D(1:10:3)  ! elements every 3 indices
    print *, array1D(2:10:4)  ! elements every 4 indices starting at the second index
    print *, array1D(10:1:-1) ! elements in reverse (start_index, end_index, increment)

    print *, array2D(:, 1)  ! first column
    
end program main