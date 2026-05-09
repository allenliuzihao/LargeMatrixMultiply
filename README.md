# Large Matrix Multiply

A high-performance C++ application for matrix multiplication operations, designed for deep learning and graphics computations.

## Overview

This project implements efficient large-scale matrix multiplication algorithms using sparse matrix formats including CSC and CSR. The implementation is part of the DeepLearningGraphics project suite and provides foundational operations for matrix computations in scientific computing and machine learning applications.

## Requirements

- **Windows 10 or later**
- **Visual Studio 2022** (with MSVC v145 toolset)
- **C++20 standard or later**

## Supported Platforms

- Win32 (x86)
- x64 (AMD64)

## Build Instructions

### Using Visual Studio 2022

1. Open `LargeMatrixMultiply.slnx` in Visual Studio 2022
2. Select your desired configuration:
   - **Debug**: For development and debugging with full debug information
   - **Release**: For optimized builds with whole program optimization
3. Select your target platform:
   - **Win32**: 32-bit x86 architecture
   - **x64**: 64-bit AMD64 architecture
4. Build the project using **Build > Build Solution** (Ctrl+Shift+B)

### Command Line Build

```bash
# Build Release x64
msbuild LargeMatrixMultiply.vcxproj /p:Configuration=Release /p:Platform=x64

# Build Debug x64
msbuild LargeMatrixMultiply.vcxproj /p:Configuration=Debug /p:Platform=x64
```

## Project Structure

```
LargeMatrixMultiply/
├── LargeMatrixMultiply.slnx        # Visual Studio solution
├── LargeMatrixMultiply.vcxproj     # Project configuration
├── LargeMatrixMultiply.vcxproj.filters
├── README.md
└── Source files (to be added)
```

## Features

- **High-Performance**: Optimized for modern multi-core CPU architectures
- **Scalable**: Efficient handling of large matrices
- **Cross-Platform Support**: Both 32-bit and 64-bit builds
- **Modern C++**: Utilizes C++20 features and standards

## Build Configuration Details

### Common Compiler Options

- **Language Standard**: C++20
- **Character Set**: Unicode
- **Console Application**: Targets Windows console subsystem
- **Security**: SDL (Security Development Lifecycle) checks enabled

### Compiler Warnings

- Warning Level: Level 3 (detects most potential issues)

### Optimization Flags

- **Release Build**: 
  - Whole Program Optimization (WPO)
  - Function-Level Linking
  - Intrinsic Functions enabled

## Usage

[Add usage instructions and examples here once implementation is complete]

## Contributing

Guidelines for contributing to this project:

1. Ensure code follows modern C++ best practices
2. Maintain C++20 standard compatibility
3. Test on both Win32 and x64 platforms
4. Build with both Debug and Release configurations

## License

[Add license information here]

## Contact

Part of the DeepLearningGraphics project suite.

---

**Note**: This README will be updated as the project develops and implementation details become available.
