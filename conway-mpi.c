#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#define HALO_MSG (0)

typedef unsigned long long ticks;

// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

// CPU Conway step for a local (s x s) block with 1-cell halo border.
// Layout: (s+2) x (s+2), data cells at [1..s][1..s].
static void compute_cpu(bool *in, bool *out, int s)
{
  int stride = s + 2;
  for (int y = 1; y <= s; ++y) {
    for (int x = 1; x <= s; ++x) {
      int alive =
        in[(y-1)*stride + (x-1)] + in[(y-1)*stride + x] + in[(y-1)*stride + (x+1)] +
        in[ y   *stride + (x-1)]                         + in[ y   *stride + (x+1)] +
        in[(y+1)*stride + (x-1)] + in[(y+1)*stride + x] + in[(y+1)*stride + (x+1)];
      bool cur = in[y*stride + x];
      out[y*stride + x] = (cur && (alive == 2 || alive == 3)) || (!cur && alive == 3);
    }
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int world_sz, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if(argc != 5) {
    fprintf(stderr, "USAGE: %s <InputFilePath> <OutputDirectory> <Steps> <WriteSteps>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int rc;
  char* inputFile = argv[1];
  MPI_File inFile;
  char* outputDir = argv[2];
  MPI_File outFile;
  int numSteps = atoi(argv[3]);
  bool writeSteps = atoi(argv[4]);

  MPI_Request reqs[8];
  MPI_Status status[8];
  rc = MPI_File_open(MPI_COMM_WORLD, inputFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &inFile);

  int dims[2];
  if(world_rank == 0) {
    MPI_File_read(inFile, dims, 2, MPI_INT, status);
    MPI_Get_count(status, MPI_INT, &rc);
  }
  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

  int w = dims[0];
  int h = dims[1];
  int numElements = w * h;
  int blockSz = numElements / world_sz;
  int s = sqrt(blockSz);

  MPI_Datatype column_type;
  MPI_Type_vector(
      s,
      1,
      s + 2,
      MPI_C_BOOL,
      &column_type
  );
  MPI_Type_commit(&column_type);

  int blockW = w / s;
  int blockH = h / s;

  int memSize = (s + 2) * (s + 2) * sizeof(bool);
  bool* local = malloc(memSize);
  bool* swap  = malloc(memSize);

  int process_grid_col = s * (world_rank % blockW);
  int process_grid_row = s * (world_rank / blockW);
  bool top    = process_grid_row < s;
  bool bottom = process_grid_row + s >= h;
  bool left   = process_grid_col < s;
  bool right  = process_grid_col + s >= w;
  int top_rank    = top    ? MPI_PROC_NULL : world_rank - blockW;
  int bottom_rank = bottom ? MPI_PROC_NULL : world_rank + blockW;
  int left_rank   = left   ? MPI_PROC_NULL : world_rank - 1;
  int right_rank  = right  ? MPI_PROC_NULL : world_rank + 1;

  int offset = (2 * sizeof(int));
  int i = (process_grid_row * w) + process_grid_col - 1;
  int len = s + 2;
  if(left) {
    for(int j = 0; j < s + 2; ++j) {
      local[j * (s + 2)] = 0;
      swap[j * (s + 2)] = 0;
    }
    --len;
    ++i;
  }
  if(right) {
    for(int j = 0; j < s + 2; ++j) {
      local[(j * (s + 2)) + (s + 1)] = 0;
      swap[(j * (s + 2)) + (s + 1)] = 0;
    }
    --len;
  }
  if(top) {
    memset(local, 0, (s + 2) * sizeof(bool));
    memset(swap,  0, (s + 2) * sizeof(bool));
  }
  if(bottom) {
    memset(local + ((s + 1) * (s + 2)), 0, (s + 2) * sizeof(bool));
    memset(swap  + ((s + 1) * (s + 2)), 0, (s + 2) * sizeof(bool));
  }

  for(int k = 1; k <= s; ++k) {
    bool* start = local + (k * (s + 2));
    if(left) { ++start; }
    MPI_File_read_at_all(inFile, offset + ((i + (w * (k - 1))) * sizeof(bool)), start, len, MPI_C_BOOL, MPI_STATUS_IGNORE);
  }
  MPI_File_close(&inFile);

  ticks comm_ticks = 0, compute_ticks = 0, io_ticks = 0;
  ticks t0;

  ticks start_t = getticks();
  for(int j = 1; j <= numSteps; ++j) {

    t0 = getticks();
    MPI_Isend(local + (s + 2) + s, 1, column_type, right_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 0);
    MPI_Irecv(local + (s + 2) + (s + 1), 1, column_type, right_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 1);

    MPI_Isend(local + (s + 2) + 1, 1, column_type, left_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 2);
    MPI_Irecv(local + (s + 2) + 0, 1, column_type, left_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 3);

    MPI_Waitall(4, reqs, status);

    MPI_Isend(local + (s * (s + 2)), s + 2, MPI_C_BOOL, bottom_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 0);
    MPI_Irecv(local, s + 2, MPI_C_BOOL, top_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 1);

    MPI_Isend(local + s + 2, s + 2, MPI_C_BOOL, top_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 2);
    MPI_Irecv(local + ((s + 1) * (s + 2)), s + 2, MPI_C_BOOL, bottom_rank, HALO_MSG, MPI_COMM_WORLD, reqs + 3);

    MPI_Waitall(4, reqs, status);
    comm_ticks += getticks() - t0;

    t0 = getticks();
    compute_cpu(local, swap, s);
    compute_ticks += getticks() - t0;

    bool* tmp = local;
    local = swap;
    swap = tmp;

    if(writeSteps || j == numSteps) {
      t0 = getticks();
      char outputFileName[strlen(outputDir) + 17];
      snprintf(outputFileName, sizeof(outputFileName), "%s/step_%d", outputDir, j);
      MPI_File_open(MPI_COMM_WORLD, outputFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);

      if(world_rank == 0) {
        MPI_File_write_at(outFile, 0, dims, 2, MPI_INT, MPI_STATUS_IGNORE);
      }

      for(int k = 1; k <= s; ++k) {
        bool* start = local + (k * (s + 2)) + 1;
        MPI_Offset byte_offset = offset + ((process_grid_row * w) + process_grid_col + (w * (k - 1))) * sizeof(bool);
        MPI_File_write_at_all(outFile, byte_offset, start, s, MPI_C_BOOL, MPI_STATUS_IGNORE);
      }
      MPI_File_close(&outFile);
      io_ticks += getticks() - t0;
    }
  }
  ticks end_t = getticks();
  double total_time   = (double)(end_t  - start_t)  / 512000000.0;
  double comm_time    = (double)comm_ticks            / 512000000.0;
  double compute_time = (double)compute_ticks         / 512000000.0;
  double io_time      = (double)io_ticks              / 512000000.0;
  printf("Rank %2d | Total: %8.4f s | Comm: %8.4f s (%5.1f%%) | Compute: %8.4f s (%5.1f%%) | IO: %8.4f s (%5.1f%%)\n",
    world_rank, total_time,
    comm_time,    100.0 * comm_time    / total_time,
    compute_time, 100.0 * compute_time / total_time,
    io_time,      100.0 * io_time      / total_time);

  free(local);
  free(swap);

  MPI_Finalize();
  return 0;
}
