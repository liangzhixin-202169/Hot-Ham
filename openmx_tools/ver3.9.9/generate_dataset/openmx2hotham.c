#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "read_scfout.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
  static int ct_AN, h_AN, Gh_AN, i, j, TNO1, TNO2;
  static int spin, Rn, myid;
  static double *a;
  static FILE *fp;

  double Ebond[30], Es, Ep;

  // int output_content = atoi(argv[2]);
  bool ham_flag = false;
  bool olp_flag = false;
  bool rr_flag = false;
  for (i = 2; i < argc; i++)
  {
    if (strcmp(argv[i], "ham") == 0)
    {
      ham_flag = true;
    }
    else if (strcmp(argv[i], "olp") == 0)
    {
      olp_flag = true;
    }
    else if (strcmp(argv[i], "rr") == 0)
    {
      rr_flag = true;
    }
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  read_scfout(argv);

  printf("atomnum=%i\n", atomnum);
  printf("SpinP_switch=%i\n", SpinP_switch);

  // Hks[spin][ct_AN][h_AN][i][j]
  if (ham_flag)
  {
    for (spin = 0; spin <= SpinP_switch; spin++)
    {
      printf("\n\nKohn-Sham Hamiltonian spin=%i\n", spin);
      for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
      {
        TNO1 = Total_NumOrbs[ct_AN];
        for (h_AN = 0; h_AN <= FNAN[ct_AN]; h_AN++)
        {
          Gh_AN = natn[ct_AN][h_AN];
          Rn = ncn[ct_AN][h_AN];
          TNO2 = Total_NumOrbs[Gh_AN];
          // printf("global index=%i  local index=%i (global=%i, Rn=%i)\n",
          //         ct_AN,h_AN,Gh_AN,Rn);
          printf("Block: %i %i %i %i %i %i %i\n", ct_AN, Gh_AN, atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3], TNO1, TNO2);
          for (i = 0; i < TNO1; i++)
          {
            for (j = 0; j < TNO2; j++)
            {
              printf("%15.8e ", Hks[spin][ct_AN][h_AN][i][j]);
            }
            printf("\n");
          }
        }
      }
    }

    // iHks[spin][ct_AN][h_AN][i][j]
    if (SpinP_switch == 3)
    {

      for (spin = 0; spin <= 2; spin++)
      {
        printf("\n\niHks: Kohn-Sham Hamiltonian spin=%i\n", spin);
        for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
        {
          TNO1 = Total_NumOrbs[ct_AN];
          for (h_AN = 0; h_AN <= FNAN[ct_AN]; h_AN++)
          {
            Gh_AN = natn[ct_AN][h_AN];
            Rn = ncn[ct_AN][h_AN];
            TNO2 = Total_NumOrbs[Gh_AN];
            // printf("global index=%i  local index=%i (global=%i, Rn=%i)\n",
            //       ct_AN,h_AN,Gh_AN,Rn);
            printf("Block: %i %i %i %i %i %i %i\n", ct_AN, Gh_AN, atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3], TNO1, TNO2);
            for (i = 0; i < TNO1; i++)
            {
              for (j = 0; j < TNO2; j++)
              {
                printf("%15.8e ", iHks[spin][ct_AN][h_AN][i][j]);
              }
              printf("\n");
            }
          }
        }
      }
    }
  }

  // OLP[ct_AN][h_AN][i][j]
  if (olp_flag)
  {
    printf("\n\nOverlap matrix\n");
    for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
    {
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN = 0; h_AN <= FNAN[ct_AN]; h_AN++)
      {
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        // printf("global index=%i  local index=%i (global=%i, Rn=%i)\n",
        //         ct_AN,h_AN,Gh_AN,Rn);
        printf("Block: %i %i %i %i %i %i %i\n", ct_AN, Gh_AN, atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3], TNO1, TNO2);
        for (i = 0; i < TNO1; i++)
        {
          for (j = 0; j < TNO2; j++)
          {
            printf("%15.8e ", OLP[ct_AN][h_AN][i][j]);
          }
          printf("\n");
        }
      }
    }
  }

  if (rr_flag)
  {
    printf("\n\nOverlap x matrix\n");
    for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
    {
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN = 0; h_AN <= FNAN[ct_AN]; h_AN++)
      {
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        printf("Block: %i %i %i %i %i %i %i\n", ct_AN, Gh_AN, atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3], TNO1, TNO2);
        for (i = 0; i < TNO1; i++)
        {
          for (j = 0; j < TNO2; j++)
          {
            printf("%15.8e ", OLPpo[0][0][ct_AN][h_AN][i][j]);
          }
          printf("\n");
        }
      }
    }

    printf("\n\nOverlap y matrix\n");
    for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
    {
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN = 0; h_AN <= FNAN[ct_AN]; h_AN++)
      {
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        printf("Block: %i %i %i %i %i %i %i\n", ct_AN, Gh_AN, atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3], TNO1, TNO2);
        for (i = 0; i < TNO1; i++)
        {
          for (j = 0; j < TNO2; j++)
          {
            printf("%15.8e ", OLPpo[1][0][ct_AN][h_AN][i][j]);
          }
          printf("\n");
        }
      }
    }

    printf("\n\nOverlap z matrix\n");
    for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
    {
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN = 0; h_AN <= FNAN[ct_AN]; h_AN++)
      {
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        printf("Block: %i %i %i %i %i %i %i\n", ct_AN, Gh_AN, atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3], TNO1, TNO2);
        for (i = 0; i < TNO1; i++)
        {
          for (j = 0; j < TNO2; j++)
          {
            printf("%15.8e ", OLPpo[2][0][ct_AN][h_AN][i][j]);
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
