# Copilot Instructions

## Project Guidelines
- User's C++ project should use std::vector for large data allocations in functions instead of std::array to avoid stack overflow. For test cases with large arrays (>10KB), allocate on heap, not stack.