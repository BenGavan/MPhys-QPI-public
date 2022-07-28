
files = main.f90 hamiltonian_2_band.f90

build: main.f90
	gfortran $(files) -o main

run: main.f90
	gfortran $(files) -o main
	./main

clean:
	rm *.mod main