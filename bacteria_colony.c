#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <mpi.h>

static int ROWS, COLS;
static int generations;

static void generate_output_filenames(const char *input_file, char *output_serial_file, char *output_parallel_file)
{
    char base_name[64];
    strcpy(base_name, input_file);

    char *dot_position = strrchr(base_name, '.');
    if (dot_position != NULL)
        *dot_position = '\0';

    strcpy(output_serial_file, base_name);
    strcat(output_serial_file, "_serial_out.txt");

    strcpy(output_parallel_file, base_name);
    strcat(output_parallel_file, "_parallel_out.txt");
}

static void print_grid(uint8_t *grid)
{
    for (int row = 0; row < ROWS; row++) 
    {
        for (int col = 0; col < COLS; col++)
            printf("%c", grid[row*COLS + col] ? 'X' : '.');
        printf("\n");
    }
}

static void save_grid_to_file(const char *filename, uint8_t *grid)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        printf("Failed to open output file \n");
        exit(-1);
    }

    fprintf(file, "%d %d\n", ROWS, COLS);
    for (int row = 0; row < ROWS; row++) 
    {
        for (int col = 0; col < COLS; col++)
            fputc(grid[row*COLS + col] ? 'X' : '.', file);
        fputc('\n', file);
    }
    fclose(file);
}

static void load_grid_from_file(const char *filename, uint8_t **out_grid)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open input file\n");
        exit(-1);
    }
    fscanf(file, "%d %d", &ROWS, &COLS);

    int elements = ROWS * COLS;
    *out_grid = (uint8_t*)malloc(elements);
    if (!*out_grid) 
    {
        printf("Memory allocation failed for grid\n");
        fclose(file);
        exit(-1);
    }

    int c, i = 0;
    while (((c = fgetc(file)) != EOF) && (i < elements))
    {
        if (c == 'X' || c == '.')   
            (*out_grid)[i++] = (c == 'X');
    }
    if (i < elements)
    {
        printf("Failed to read whole grid\n");
        fclose(file);
        exit(-1);
    }
    fclose(file);
}

static void swap_buffers(uint8_t **a, uint8_t **b)
{
    uint8_t *tmp = *a;
    *a = *b;
    *b = tmp;
}

static int count_neighbors(uint8_t *grid, int rows, int cols, int row, int col)
{
    static const int drow[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    static const int dcol[8] = {-1,  0,  1, -1, 1, -1, 0, 1};

    int neighbors = 0; 
    for (int i = 0; i < 8; i++)
    {
        int nrow = row + drow[i];
        int ncol = col + dcol[i];
        if ((nrow >= 0 && nrow < rows) && (ncol >= 0 && ncol < cols))
        {
            neighbors += grid[nrow * cols + ncol];
        }
    }
    return neighbors;
}

static void compute_next_generation(uint8_t *current, uint8_t *next, int rows, int cols,
                                   int row_start, int row_end)
{
    for (int row = row_start; row <= row_end; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int neighbors = count_neighbors(current, rows, cols, row, col);
            uint8_t is_alive = current[row * cols + col];
            next[row * cols + col] = is_alive ? (neighbors == 2 || neighbors == 3) : (neighbors == 3);
        }
    }
}

static void simulate_serial(uint8_t *initial_grid, uint8_t **out_final_grid)
{
    int elements = ROWS * COLS;
    uint8_t *current_gen = (uint8_t*)malloc(elements);
    uint8_t *next_gen    = (uint8_t*)malloc(elements);
    if (!current_gen || !next_gen)
    {
        printf("Memory allocation failed for serial grids\n");
        exit(-1);
    }
    memcpy(current_gen, initial_grid, elements);

#ifdef DEBUG
    printf("Initial:\n");
    print_grid(current_gen);
#endif

    for (int gen = 0; gen < generations; gen++)
    {
        compute_next_generation(current_gen, next_gen, ROWS, COLS, 0, ROWS - 1);
        swap_buffers(&current_gen,&next_gen);

#ifdef DEBUG
        printf("After gen %d:\n", gen + 1);
        print_grid(current_gen);
#endif
    }
    free(next_gen);
    *out_final_grid = current_gen;
}

static void exchange_halo_rows(uint8_t *local_grid, int local_rows, int rank, int size)
{
    int up = (rank == 0) ? MPI_PROC_NULL : (rank - 1);
    int down = (rank == size - 1) ? MPI_PROC_NULL : (rank + 1);

    MPI_Sendrecv(local_grid + COLS, COLS, MPI_UINT8_T, up, 0,
                 local_grid, COLS, MPI_UINT8_T, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(local_grid + local_rows * COLS, COLS, MPI_UINT8_T, down, 1,
                 local_grid + (local_rows + 1) * COLS, COLS, MPI_UINT8_T, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static void simulate_parallel(uint8_t *initial_grid, int rank, int size, uint8_t **out_final_grid)
{
    int base_rows = ROWS / size;
    int extra_rows = ROWS % size;

    int local_rows = base_rows + (rank < extra_rows ? 1 : 0);

    int elements = (local_rows + 2) * COLS;
    uint8_t *local_current_gen = (uint8_t *)calloc((size_t)elements, sizeof(uint8_t));
    uint8_t *local_next_gen    = (uint8_t *)calloc((size_t)elements, sizeof(uint8_t));
    if (!local_current_gen || !local_next_gen)
    {
        if (rank == 0)
            printf("Memory allocation failed for local_current_gen / local_next_gen\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *send_counts = NULL;
    int *send_offsets = NULL;
    uint8_t *send_grid = NULL;

    if (rank == 0)
    {
        send_counts = (int *)malloc((size_t)size * sizeof(int));
        send_offsets = (int *)malloc((size_t)size * sizeof(int));
        if (!send_counts || !send_offsets)
        {
            printf("Memory allocation failed for send_counts / send_offsets\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int row = 0; row < size; row++)
        {
            int rows_for_rank  = base_rows + (row < extra_rows ? 1 : 0);
            int start_for_rank = row * base_rows + (row < extra_rows ? row : extra_rows);
            send_counts[row]  = rows_for_rank * COLS;
            send_offsets[row] = start_for_rank * COLS;
        }

        send_grid = initial_grid;
    }

    MPI_Scatterv(send_grid, send_counts, send_offsets, MPI_UINT8_T,
                 local_current_gen + COLS, local_rows * COLS, MPI_UINT8_T,
                 0, MPI_COMM_WORLD);

    uint8_t *final_grid_parallel = NULL;
    if (rank == 0)
    {
        final_grid_parallel = (uint8_t *)malloc((size_t)ROWS * (size_t)COLS);
        if (!final_grid_parallel)
        {
            printf("Memory allocation failed for final_grid_parallel\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int gen = 0; gen < generations; gen++)
    {
        exchange_halo_rows(local_current_gen, local_rows, rank, size);
        compute_next_generation(local_current_gen, local_next_gen, local_rows + 2, COLS, 1, local_rows);
        swap_buffers(&local_current_gen,&local_next_gen);
    }

    MPI_Gatherv(local_current_gen + COLS, local_rows * COLS, MPI_UINT8_T,
                final_grid_parallel, send_counts, send_offsets, MPI_UINT8_T,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        *out_final_grid = final_grid_parallel;
        free(send_counts);
        free(send_offsets);
    }

    free(local_current_gen);
    free(local_next_gen);
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) 
    {
        if (rank == 0)
            printf("Usage: %s input.txt G\n", argv[0]);
        MPI_Finalize();
        return 0;
    }
    const char *input_filename = argv[1];
    generations = atoi(argv[2]);
    if (generations < 1)
    {
        if (rank == 0)
            printf("Invalid number of generations\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    uint8_t *initial_grid = NULL;
    char output_serial_file[64];
    char output_parallel_file[64];
    if (rank == 0)
    {
        load_grid_from_file(input_filename, &initial_grid);
        printf("Rows=%d Cols=%d Generations=%d\n", ROWS, COLS, generations);
        generate_output_filenames(input_filename, output_serial_file, output_parallel_file);
    }

    MPI_Bcast(&ROWS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&COLS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&generations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double serial_duration;
    uint8_t *final_grid_serial = NULL;
    if (rank == 0)
    {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        simulate_serial(initial_grid, &final_grid_serial);
        clock_gettime(CLOCK_MONOTONIC, &end);
        serial_duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        save_grid_to_file(output_serial_file, final_grid_serial);
        printf("Serial: %.6f s\n", serial_duration);
    }

    uint8_t *final_grid_parallel = NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    simulate_parallel(initial_grid, rank, size, &final_grid_parallel);
    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();

    double local_time = stop - start;
    double parallel_duration;
    MPI_Reduce(&local_time, &parallel_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        save_grid_to_file(output_parallel_file, final_grid_parallel);
        printf("Parallel: %.6f s\n", parallel_duration);

        int elements = ROWS * COLS;
        int grids_are_identical = (memcmp(final_grid_serial, final_grid_parallel, elements) == 0);
        if (!grids_are_identical) printf("Serial and parallel are not the same\n");

        printf("Speedup: %.2fx\n", serial_duration / parallel_duration);

        free(initial_grid);
        free(final_grid_serial);
        free(final_grid_parallel);
    }
    
    MPI_Finalize();
    return 0;
}