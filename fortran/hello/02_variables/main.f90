program variables
    implicit none

    integer :: amount
    integer :: testInt
    integer :: testInt2
    real :: pi
    complex :: frequency
    character :: initial
    logical :: isOkay

    complex :: frequency2
    complex :: frequency3

    real :: square

    amount = 10
    testInt = -10
    pi = 3.1415927
    frequency = (1.1, -2.6)
    initial = 'B'
    isOkay = .false.

    print *, amount
    print *, testInt
    print *, testInt2
    print *, pi
    print *, initial

    print *, isOkay
    isOkay = .true.
    print *, isOkay

    print *, frequency

    print *, 'The value of amount (integer) is: ', amount
    print *, 'The value of frequency (complex) is:', frequency
    print *, 'Value of isOkay (logical (boolean)):', isOkay

    frequency2 = (1, 1)
    frequency3 = frequency + frequency2

    print *, frequency, ' + ', frequency2, ' = '
    print *, 'sum of frequencies: ', frequency3

    frequency3 = frequency * frequency2
    print *, frequency, ' * ', frequency2, ' = ', frequency3

    square =  2.0**2.0
    print *, frequency, ' * ', frequency2, ' = ', frequency3

    print *, "square = ", square


end program variables