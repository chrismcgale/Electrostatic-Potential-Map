#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <random>
#include <iomanip>

#define MAX_ATOMS 1032


// Must be <= 16kb
__constant__ float atoms[1032 * 4];


// Scatter approach requires atomic operations, gather approach is more parrallelizable

void __global__ c_energy(float* energy_grid, dim3 grid, float grid_spacing, float z, int num_atoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int atom_arr_dim = num_atoms * 4;
    int k = z / grid_spacing;

    float x = grid_spacing * (float) i;
    float y = grid_spacing * (float) j;
    
    float energy = 0.0f;

    for (int n = 0; n < atom_arr_dim; n += 4) {
        float dx = x - atoms[n];
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
    }

    energy_grid[grid.x*grid.y*k + grid.x*j + i] += energy;
}

void __host__ generate_test_atoms() {
    std::ofstream file("atoms.txt"); // Output file
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_xyz(0.0, 100);
    std::uniform_real_distribution<float> dis_potential(0.0, 10.0);

    if (file.is_open()) {
        for (int i = 0; i < 100; ++i) {
            float x = dis_xyz(gen);
            float y = dis_xyz(gen);
            float z = dis_xyz(gen);
            float potential = dis_potential(gen);

            file << std::fixed << std::setprecision(2)
                 << x << " " << y << " " << z << " " << potential << "\n";
        }
        file.close();
        std::cout << "Data file generated successfully.\n";
    } else {
        std::cerr << "Unable to create the file.\n";
    }
}

int __host__ main() {
    #ifdef __TEST__
    generate_test_atoms();
    #endif

    std::ifstream file("atoms.txt"); // Replace "input.txt" with your file name
    std::string line;
    float h_atoms[1032 * 4];
    int num_atoms = 0;
    // Assume all atoms are within this 128 x 128 x 128 cube
    int size_x = 128, size_y = 128, size_z = 128;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z, potential;
        if (iss >> x >> y >> z >> potential) {

            h_atoms[num_atoms * 4] = x;
            h_atoms[num_atoms * 4 + 1] = y;
            h_atoms[num_atoms * 4 + 2] = z;
            h_atoms[num_atoms * 4 + 3] = potential;
            num_atoms += 1;
        }
    }

    cudaMemcpyToSymbol(atoms, h_atoms, num_atoms * sizeof(float));

    float* h_energy_grid = new float[size_x * size_y * size_z], *d_energy_grid;
    dim3 grid(size_x, size_y, size_z);
    float grid_spacing = 0.5f;


    cudaMalloc((void**)&d_energy_grid, size_x * size_y * size_z * sizeof(float));

    cudaMemcpy(d_energy_grid, h_energy_grid, size_x * size_y * size_z * sizeof(float), cudaMemcpyHostToDevice);

    for (int z = 0; z < size_z; z++) {
        c_energy<<<dim3(4, 4), dim3(32, 32)>>>(d_energy_grid, grid, grid_spacing, z, num_atoms);
    }

    cudaFree(d_energy_grid);

    delete [] h_energy_grid;
}
