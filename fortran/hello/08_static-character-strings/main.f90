program main
    implicit none

    character(len=3) :: first_name
    character(len=5) :: last_name
    character(len=9) :: full_name
    character(len=8) :: appended_name
    character(len=1) :: break_name

    break_name = 'Hi'

    print *, break_name

    first_name = 'Ben'
    last_name = 'Gavan'
    full_name = first_name//' '//last_name
    appended_name = first_name//last_name

    print *, full_name
    print *, appended_name

end program main