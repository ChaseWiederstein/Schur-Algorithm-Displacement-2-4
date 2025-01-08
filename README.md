# Schur-Algorithm-Displacement-2-4

## Executive Summary
This project implements and validates the Schur algorithm for computing the Cholesky factor of symmetric positive definite (SPD) matrices and Toeplitz matrices with displacement ranks 2 and 4. The algorithms explore the relationship between matrix structure and algorithmic performance, focusing on numerical accuracy and computational stability. Key tasks include generating test matrices, validating the Cholesky factors, and analyzing the performance on both structured and general SPD matrices.

## Key Sections
1. **Rank 2 Displacement Validation**:
  - Implementation and validation of the Schur algorithm for SPD Toeplitz and general SPD matrices.
  - Analysis of errors for reconstructed matrices and Cholesky factors.
  - Exploration of hyperbolic rotation failures for indefinite matrices.
2. **Rank 4 Displacement Validation**:
  - Implementation of the Schur algorithm for least squares problems involving SPD Gram matrices.
  - Comparison of rectangular and square Toeplitz matrices and their respective performances.
  - Evaluation of displacement errors, normal equation errors, and Cholesky factorization errors.
3. **Conclusions**: Summarizes the algorithm's performance, emphasizing its strengths with Toeplitz structures and limitations with general SPD matrices.

## Project Files
- **`rank2.py`**: Implements the Schur algorithm for SPD matrices with rank 2 displacement. Includes validation for SPD Toeplitz and general SPD matrices.
- **`rank4.py`**: Implements the Schur algorithm for least squares problems with rank 4 displacement. Validates rectangular and square Toeplitz matrices.
- **`writeup.pdf`**: A detailed report covering methodology, validation results, and analysis of the Schur algorithm’s performance for displacement ranks 2 and 4.
