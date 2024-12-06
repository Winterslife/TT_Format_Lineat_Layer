./TestTTSimulation 4 1 16 16 16 1 8 8 8 2 8 8 4 2


# Create a build directory for float precision
mkdir build_float && cd build_float

# Configure with float precision
cmake .. -DMM_DATA_TYPE=float

# Build
make TestTTPrecision

# Create a build directory for float precision
mkdir build_float && cd build_float

# Configure with float precision
cmake .. -DMM_DATA_TYPE=float

# Build
make TestTTPrecision

# Run with different configurations
./TestTTPrecision 3 16  # Small configuration
./TestTTPrecision 4 32  # Medium configuration
./TestTTPrecision 5 64  # Large configuration