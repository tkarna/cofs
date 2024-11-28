all: invert plot

CASE        ?= Constant # Regions IndependentPointsScheme NodalFreedom
STATIONS    = stationA stationB stationC stationD stationE
PARALLEL    ?= 0 # Default to serial execution

invert:
ifeq ($(PARALLEL), 0)
	python3 inverse_problem.py --case $(CASE) --no-consistency-test --no-taylor-test
else
	mpiexec -np $(PARALLEL) python inverse_problem.py --case $(CASE) --no-consistency-test --no-taylor-test;
endif

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
