program main
    implicit none

    integer :: stat
    integer :: lines
    character(len=200) :: line
    character(len=10) :: word

    integer :: i

    lines = 0

    open (1, file='data.dat', status='old')  ! can not have unit=6

    do
        read(1, '(A)', IOSTAT=stat) line
        if (IS_IOSTAT_END(stat)) exit
        lines = lines + 1
        print *, line
        do i = 1, 200
            print *, line(i:i)
        end do
    end do

    close (1)

    print *, 'finshed reading ', lines, ' lines'
    print *, 'test'(2:2)
end program main