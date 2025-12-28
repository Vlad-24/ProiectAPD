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

static int count_neighbors(uint8_t *grid, int row, int col)
{
    static const int drow[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    static const int dcol[8] = {-1,  0,  1, -1, 1, -1, 0, 1};

    int neighbors = 0; 
    for (int i = 0; i < 8; i++)
    {
        int nrow = row + drow[i];
        int ncol = col + dcol[i];
        if ((nrow >= 0 && nrow < ROWS) && (ncol >= 0 && ncol < COLS))
        {
            neighbors += grid[nrow * COLS + ncol];
        }
    }
    return neighbors;
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
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int neighbors = count_neighbors(current_gen, row, col);
                uint8_t is_alive = current_gen[row * COLS + col];
                next_gen[row * COLS + col] = is_alive ? (neighbors == 2 || neighbors == 3) : (neighbors == 3);
            }
        }
        uint8_t *temp = current_gen;
        current_gen = next_gen;
        next_gen = temp;

#ifdef DEBUG
        printf("After gen %d:\n", gen + 1);
        print_grid(current_gen);
#endif
    }
    free(next_gen);
    *out_final_grid = current_gen;
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
        exit(-1);
    }

    if (rank == 0)
    {
        uint8_t *initial_grid = NULL;
        load_grid_from_file(input_filename, &initial_grid);

        printf("Rows=%d Cols=%d Generations=%d\n", ROWS, COLS, generations);

        char output_serial_file[64];
        char output_parallel_file[64];
        generate_output_filenames(input_filename, output_serial_file, output_parallel_file);

        struct timespec start, end;

        uint8_t *final_grid_serial = NULL;
        clock_gettime(CLOCK_MONOTONIC, &start);
        simulate_serial(initial_grid, &final_grid_serial);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double serial_duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        save_grid_to_file(output_serial_file, final_grid_serial);
        printf("Serial: %.6f s\n", serial_duration);

#ifdef DEBUG
        printf("\nFinal grid (serial):\n");
        print_grid(final_grid_serial);
#endif
        free(initial_grid);
        free(final_grid_serial);
    }

    MPI_Finalize();
    return 0;
}