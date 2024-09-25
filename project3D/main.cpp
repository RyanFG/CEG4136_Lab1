#define USE_CUDA

#include <GL/glut.h>
#include <vector>
#include <random>    // For std::shuffle and std::mt19937
#include <algorithm> // For std::shuffle
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

#define N 1000  // Grid size
#if N > 1024
#error "Current implemantation forbids grids bigger than 1024x1024."
#endif
#define BURN_DURATION 5000   // Tree burning duration in milliseconds (5 seconds)
#define FIRE_START_COUNT 100  // Initial number of fire locations
#if FIRE_START_COUNT > 1024
#error "Current implemantation forbids more than 1024 initial fires."
#endif

// Using vectors to manage memory
std::vector<std::vector<int>> forest(N, std::vector<int>(N, 0));
std::vector<std::vector<int>> burnTime(N, std::vector<int>(N, 0));

int simulationDuration = 60000;  // Simulation duration (60 seconds)
int startTime = 0;   // Start time in milliseconds
int elapsedTime = 0;  // Elapsed time
float spreadProbability = 0.3f;  // Probability that fire spreads to a neighboring tree

bool isPaused = false; // Pause indicator
int pauseStartTime = 0;   // Start time of pause

float zoomLevel = 1.0f; // Zoom level
float offsetX = 0.0f, offsetY = 0.0f;  // Horizontal and vertical offset for movement
float moveSpeed = 0.05f; // View movement speed

bool dragging = false;  // Mouse drag indicator
int lastMouseX, lastMouseY;  // Last mouse position when clicked

#ifdef USE_CUDA
int* forest_GPU;
int* burnTime_GPU;
int* newForest_GPU;

curandState* dev_curand_states;

__global__ void setup_kernel(curandState* state, uint64_t seed) {
    int gid = blockIdx.x * N + threadIdx.x;

    curand_init(seed, gid, 0, &state[gid]);
}

__global__ void initializeTree(curandState* globalState, int* forest_GPU, int* burnTime_GPU) {
    int gid = blockIdx.x * N + threadIdx.x;

    forest_GPU[gid] = curand_uniform(globalState + gid) < 0.5 ? 0 : 1;  // 50% trees (1), 50% empty space (0)
    burnTime_GPU[gid] = 0;         // No tree is burning at the start
}

__global__ void igniteTree(std::pair<int, int>* availablePositions_GPU, int* forest_GPU, int* burnTime_GPU) {
    int gid = availablePositions_GPU[threadIdx.x].first * N + availablePositions_GPU[threadIdx.x].second;

    forest_GPU[gid] = 2;
    burnTime_GPU[gid] = BURN_DURATION;
}

__global__ void fireSpreading(curandState* globalState, int* forest_GPU, int* newForest_GPU, int* burnTime_GPU, bool* burntOut) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;            // current gid
    int gidL = (threadIdx.x - 1) + blockIdx.x * blockDim.x;     // left gid
    int gidR = (threadIdx.x + 1) + blockIdx.x * blockDim.x;     // right gid
    int gidU = threadIdx.x + ((blockIdx.x + 1) * blockDim.x);   // up gid
    int gidD = threadIdx.x + ((blockIdx.x - 1) * blockDim.x);   // down gid

    if (forest_GPU[gid] == 2) {
        burnTime_GPU[gid] -= 200;
        if (burnTime_GPU[gid] <= 0) {
            newForest_GPU[gid] = 3;
        }
    }
    else if (forest_GPU[gid] == 1) {
        if (forest_GPU[gidL] == 2 && threadIdx.x > 0 && (curand_normal(globalState + gidL) < 0.3f)) { //check left
            newForest_GPU[gid] = 2;
            burnTime_GPU[gid] = 5000;
        };
        if (forest_GPU[gidR] == 2 && threadIdx.x <= 999 && (curand_normal(globalState + gidR) < 0.3f)) { //check right
            newForest_GPU[gid] = 2;
            burnTime_GPU[gid] = 5000;
        };
        if (forest_GPU[gidU] == 2 && blockIdx.x > 0 && (curand_normal(globalState + gidU) < 0.3f)) { //check above
            newForest_GPU[gid] = 2;
            burnTime_GPU[gid] = 5000;
        };
        if (forest_GPU[gidD] == 2 && blockIdx.x <= 999 && (curand_normal(globalState + gidD) < 0.3f)) { // check below
            newForest_GPU[gid] = 2;
            burnTime_GPU[gid] = 5000;
        };
    }

    if (forest_GPU[gid] == 2) {
        *burntOut = false;
    }
}
#endif

// Function to initialize the forest
#ifdef USE_CUDA
void initializeForest() {
    // Initializing the forest with 50% trees
    // Now Optimized with CUDA®

    initializeTree << <N, N >> > (dev_curand_states, forest_GPU, burnTime_GPU);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < forest.size(); i++) cudaMemcpy(forest[i].data(), forest_GPU + i * N, N * sizeof(int), cudaMemcpyDeviceToHost);

    // List of available positions to start fires
    // Can maybe use threads here? optimize
    std::vector<std::pair<int, int>> availablePositions;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (forest[i][j] == 1) {   // Add positions with trees to the list
                availablePositions.push_back({ i, j });
            }
        }
    }

    // Shuffle the available positions for a more uniform distribution
    std::random_device rd;  // Random number generator based on system implementation
    std::mt19937 g(rd());  // Mersenne Twister-based pseudo-random number generator
    std::shuffle(availablePositions.begin(), availablePositions.end(), g);

    // Ignite fires uniformly across the grid
    // Now Optimized with CUDA®
    std::pair<int, int>* availablePositions_GPU;
    cudaMalloc(&availablePositions_GPU, availablePositions.size() * sizeof(std::pair<int, int>));
    cudaMemcpy(availablePositions_GPU, availablePositions.data(), availablePositions.size() * sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);

    size_t fires_to_start = availablePositions.size() < FIRE_START_COUNT ? availablePositions.size() : FIRE_START_COUNT;
    igniteTree << <1, fires_to_start >> > (availablePositions_GPU, forest_GPU, burnTime_GPU);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < forest.size(); i++) {
        cudaMemcpy(forest[i].data(), forest_GPU + i * N, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(burnTime[i].data(), burnTime_GPU + i * N, N * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(&availablePositions_GPU);

    startTime = glutGet(GLUT_ELAPSED_TIME);  // Reset start time
    elapsedTime = 0;  // Reset elapsed time
    isPaused = false; // End of pause
}
#else
void initializeForest() {
    // Initializing the forest with 50% trees
    // Create Threads here instead of a 2D for loop, use a 2D grid of threads 1000 by 1000, optimize
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            forest[i][j] = rand() % 2; // 50% trees (1), 50% empty space (0)
            burnTime[i][j] = 0;  // No tree is burning at the start
        }
    }
    // List of available positions to start fires
    // Can maybe use threads here? optimize
    std::vector<std::pair<int, int>> availablePositions;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (forest[i][j] == 1) {   // Add positions with trees to the list
                availablePositions.push_back({ i, j });
            }
        }
    }
    // Shuffle the available positions for a more uniform distribution
    std::random_device rd;  // Random number generator based on system implementation
    std::mt19937 g(rd());  // Mersenne Twister-based pseudo-random number generator
    std::shuffle(availablePositions.begin(), availablePositions.end(), g);
    // Ignite fires uniformly across the grid
    for (int fire = 0; fire < FIRE_START_COUNT && !availablePositions.empty(); fire++) {
        int fireX = availablePositions[fire].first;
        int fireY = availablePositions[fire].second;
        forest[fireX][fireY] = 2; // Ignite the tree
        burnTime[fireX][fireY] = BURN_DURATION; // Set the burn duration
    }
    startTime = glutGet(GLUT_ELAPSED_TIME);  // Reset start time
    elapsedTime = 0;  // Reset elapsed time
    isPaused = false; // End of pause
}
#endif

// OpenGL initialization function
void initGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);   // White background color
    glEnable(GL_DEPTH_TEST); // Enable depth test
}

// Function to draw the grid
void drawForest() {
    float cellSize = 2.0f / N;  // Adjusted cell size based on grid size N

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Set color based on the state of the cell
            if (forest[i][j] == 0 && burnTime[i][j] == 0) {
                glColor3f(0.8f, 0.8f, 0.8f);  // Empty space (gray)
            }
            else if (forest[i][j] == 1) {
                glColor3f(0.0f, 1.0f, 0.0f); // Tree (green)
            }
            else if (forest[i][j] == 2) {
                glColor3f(1.0f, 0.0f, 0.0f);  // Tree on fire (red)
            }
            else if (forest[i][j] == 3) {
                glColor3f(0.0f, 0.0f, 0.0f);  // Burned tree (black)
            }

            // Draw the cell
            float x = -1.0f + j * cellSize;
            float y = -1.0f + i * cellSize;
            glBegin(GL_QUADS);
            glVertex2f(x, y);
            glVertex2f(x + cellSize, y);
            glVertex2f(x + cellSize, y + cellSize);
            glVertex2f(x, y + cellSize);
            glEnd();
        }
    }
}

// Function to update the forest and fire propagation
#ifdef USE_CUDA
void updateForest() {
    if (isPaused) {  // If the simulation is paused, reset the forest after the pause
        if (glutGet(GLUT_ELAPSED_TIME) - pauseStartTime >= 3000) {
            initializeForest(); // Reset the forest after 3 seconds
        }
        return;
    }

    std::vector<std::vector<int>> newForest = forest;  // Copy the current forest

    // Now Optimized with CUDA®
    bool allBurnedOut = true; // Flag to check if all fires are out
    bool* burntOut_GPU;

    cudaMalloc(&burntOut_GPU, sizeof(bool));
    for (size_t i = 0; i < forest.size(); i++) {
        cudaMemcpy(forest_GPU + i * N, forest[i].data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(burnTime_GPU + i * N, burnTime[i].data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(newForest_GPU + i * N, newForest[i].data(), N * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(burntOut_GPU, &allBurnedOut, sizeof(bool), cudaMemcpyHostToDevice);


    fireSpreading << <N, N >> > (dev_curand_states, forest_GPU, newForest_GPU, burnTime_GPU, burntOut_GPU);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < forest.size(); i++) {
        cudaMemcpy(forest[i].data(), forest_GPU + i * N, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(burnTime[i].data(), burnTime_GPU + i * N, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(newForest[i].data(), newForest_GPU + i * N, N * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(&allBurnedOut, burntOut_GPU, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(burntOut_GPU);



    forest = newForest;  // Update the forest with the new copy

    if (allBurnedOut) {  // If all fires are out, pause the simulation
        isPaused = true;
        pauseStartTime = glutGet(GLUT_ELAPSED_TIME);
    }
}
#else
void updateForest() {
    if (isPaused) {  // If the simulation is paused, reset the forest after the pause
        if (glutGet(GLUT_ELAPSED_TIME) - pauseStartTime >= 3000) {
            initializeForest(); // Reset the forest after 3 seconds
        }
        return;
    }
    std::vector<std::vector<int>> newForest = forest;  // Copy the current forest
    bool allBurnedOut = true; // Flag to check if all fires are out
    // Optimize by using grid of threads to eval each tree seperately
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (forest[i][j] == 2) {  // If the tree is on fire
                burnTime[i][j] -= 200;  // Reduce the burning time
                // Check if the fire is out
                if (burnTime[i][j] <= 0) {
                    newForest[i][j] = 3;  // Mark the tree as burned
                }
                else {
                    // Propagation of fire to neighbors
                    if (i > 0 && forest[i - 1][j] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                        newForest[i - 1][j] = 2;
                        burnTime[i - 1][j] = BURN_DURATION;
                    }
                    if (i < N - 1 && forest[i + 1][j] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                        newForest[i + 1][j] = 2;
                        burnTime[i + 1][j] = BURN_DURATION;
                    }
                    if (j > 0 && forest[i][j - 1] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                        newForest[i][j - 1] = 2;
                        burnTime[i][j - 1] = BURN_DURATION;
                    }
                    if (j < N - 1 && forest[i][j + 1] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                        newForest[i][j + 1] = 2;
                        burnTime[i][j + 1] = BURN_DURATION;
                    }
                }
            }
            // If a tree is still burning, continue the simulation
            if (forest[i][j] == 2) {
                allBurnedOut = false;
            }
        }
    }
    forest = newForest;  // Update the forest with the new copy
    if (allBurnedOut) {  // If all fires are out, pause the simulation
        isPaused = true;
        pauseStartTime = glutGet(GLUT_ELAPSED_TIME);
    }
}
#endif

// Display function
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clear color and depth buffer
    glLoadIdentity(); // Reset the model-view matrix
    glTranslatef(offsetX, offsetY, 0.0f);  // Apply translation offset
    glScalef(zoomLevel, zoomLevel, 1.0f);   // Apply zoom
    drawForest(); // Draw the forest
    glutSwapBuffers(); // Swap buffers to display the image
}

// Function to animate the simulation
void update(int value) {
    updateForest();  // Update the forest at each cycle
    glutPostRedisplay();  // Request a new rendering
    glutTimerFunc(200, update, 0);  // Schedule the next update in 200 ms
}

// Keyboard handling for zooming and resetting
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case '+':
        zoomLevel *= 1.1f;  // Increase zoom level
        break;
    case '-':
        zoomLevel /= 1.1f;  // Decrease zoom level
        if (zoomLevel < 0.1f) zoomLevel = 0.1f;
        break;
    case 'r': // Reset key
        zoomLevel = 1.0f;  // Reset zoom and offset
        offsetX = 0.0f;
        offsetY = 0.0f;
        break;
    case 27:  // Escape key to quit
        exit(0);
    }
    glutPostRedisplay(); // Redraw the scene
}

// Arrow keys handling for moving the view
void specialKeys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP:
        offsetY += moveSpeed / zoomLevel;  // Move the view up
        break;
    case GLUT_KEY_DOWN:
        offsetY -= moveSpeed / zoomLevel;  // Move the view down
        break;
    case GLUT_KEY_LEFT:
        offsetX += moveSpeed / zoomLevel;  // Move the view left
        break;
    case GLUT_KEY_RIGHT:
        offsetX -= moveSpeed / zoomLevel;  // Move the view right
        break;
    }
    glutPostRedisplay();   // Redraw the scene
}

// Mouse handling for moving the view
void mouseMotion(int x, int y) {
    if (dragging) {
        offsetX += (x - lastMouseX) * moveSpeed / zoomLevel;  // Update horizontal offset
        offsetY -= (y - lastMouseY) * moveSpeed / zoomLevel; // Update vertical offset
        lastMouseX = x;
        lastMouseY = y;
        glutPostRedisplay(); // Redraw the scene
    }
}

// Function to handle mouse clicks
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {  // If the left mouse button is pressed
        if (state == GLUT_DOWN) {
            dragging = true;
            lastMouseX = x;
            lastMouseY = y;
        }
        else {
            dragging = false;
        }
    }
}

// Main function
int main(int argc, char** argv) {
    srand(static_cast<unsigned>(time(NULL)));  // Initialize random number generator
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Simulation de feux de forêt/Forest Fire Simulation"); // Create the OpenGL window

    initGL();
#ifdef USE_CUDA
    cudaMalloc((void**)&forest_GPU, N * N * sizeof(int));
    cudaMalloc((void**)&burnTime_GPU, N * N * sizeof(int));
    cudaMalloc((void**)&newForest_GPU, N * N * sizeof(int));
    cudaMalloc(&dev_curand_states, N * N * sizeof(curandState));
    setup_kernel << <N, N >> > (dev_curand_states, time(NULL));
#endif
    initializeForest();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutTimerFunc(200, update, 0);

    glutMainLoop();
#ifdef USE_CUDA
    cudaFree(&forest_GPU);
    cudaFree(&burnTime_GPU);
    cudaFree(&newForest_GPU);
#endif
    return 0;
}
