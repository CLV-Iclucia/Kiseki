//
// Created by creeper on 5/23/24.
//
#include <Maths/sparse-matrix-builder.h>
using namespace ksk::maths;
int main() {
  std::cout << "Input the number of rows and columns: ";
  int n, m;
  std::cin >> n >> m;
  SparseMatrixBuilder<double> builder(n, m);
  while (true) {
    std::cout << "Input command: ";
    std::string command;
    std::cin >> command;
    if (command == "add") {
      int row, col;
      double value;
      std::cin >> row >> col >> value;
      builder.addElement(row, col, value);
    } else if (command == "assemble") {
      std::cout << "Input matrix 3x3: ";
      Matrix<double, 3, 3> matrix;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          std::cin >> matrix(i, j);
      std::cout << "Input 3 indices: ";
      int i, j, k;
      std::cin >> i >> j >> k;
      builder.assemble(matrix, i, j, k);
    } else if (command == "assembleBlock") {
      std::cout << "Input matrix 6x6: ";
      Matrix<double, 6, 6> matrix;
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          std::cin >> matrix(i, j);
      std::cout << "Input 2 block indices: ";
      int i, j;
      std::cin >> i >> j;
      builder.assembleBlock<6, 3>(matrix, i, j);
    } else {
      continue;
    }
    auto matrix = builder.build();
    // convert to dense matrix
    Eigen::MatrixXd dense_matrix = matrix;
    std::cout << dense_matrix << std::endl;
  }
}