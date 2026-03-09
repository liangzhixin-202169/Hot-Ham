# Convert OpenMX results to Hot-Ham format

## Compile
Compile a executable file `openmx2hotham`:
```shell
CC="mpicc -O3 -fopenmp"
$CC openmx2hotham.c read_scfout.c -o openmx2hotham
```
## Run
If you want to convert some properties saved in `file.scfout`, you can execute:
```shell
openmx2hotham file.scfout property_0 property_1 ... > Hks.txt
```
then these properties will be saved in Hks.txt with Hot-Ham format. Now the supported properties includes:
 - Hamiltonian: `ham`
 - Overlap: `olp`
 - Real space position matrix: `rr`.