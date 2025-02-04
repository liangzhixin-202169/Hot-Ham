#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "read_scfout.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
  static int ct_AN,h_AN,Gh_AN,i,j,TNO1,TNO2;
  static int spin,Rn,myid;
  static double *a;
  static FILE *fp;

  double Ebond[30],Es,Ep;

  int output_content = atoi(argv[2]);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  read_scfout(argv);

  printf("atomnum=%i\n", atomnum);
  printf("SpinP_switch=%i\n", SpinP_switch);

  // Hks[spin][ct_AN][h_AN][i][j]
  for (spin=0; spin<=SpinP_switch; spin++){
    printf("\n\nKohn-Sham Hamiltonian spin=%i\n",spin);
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        // printf("global index=%i  local index=%i (global=%i, Rn=%i)\n",
        //         ct_AN,h_AN,Gh_AN,Rn);
        printf("Block: %i %i %i %i %i %i %i\n",ct_AN,Gh_AN,atv_ijk[Rn][1],atv_ijk[Rn][2],atv_ijk[Rn][3],TNO1,TNO2);
        for (i=0; i<TNO1; i++){
          for (j=0; j<TNO2; j++){
            printf("%10.7f ",Hks[spin][ct_AN][h_AN][i][j]);
          }
          printf("\n");
        }
      }
    }
  }

 // OLP[ct_AN][h_AN][i][j]
 if (output_content==1)
 {
  printf("\n\nOverlap matrix\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      Rn = ncn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      //printf("global index=%i  local index=%i (global=%i, Rn=%i)\n",
      //        ct_AN,h_AN,Gh_AN,Rn);
      printf("Block: %i %i %i %i %i %i %i\n",ct_AN,Gh_AN,atv_ijk[Rn][1],atv_ijk[Rn][2],atv_ijk[Rn][3],TNO1,TNO2);
      for (i=0; i<TNO1; i++){
        for (j=0; j<TNO2; j++){
          printf("%10.7f ",OLP[ct_AN][h_AN][i][j]);
        }
        printf("\n");
      }
    }
  }
 }

 // Gxyz
 // printf("\n");
 // for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
 //  printf("ct_AN=%i  %10.7f  %10.7f  %10.7f  %10.7f  %10.7f\n",ct_AN,Gxyz[ct_AN][0],Gxyz[ct_AN][1],Gxyz[ct_AN][2],Gxyz[ct_AN][3],Gxyz[ct_AN][4]);
 // }

}
