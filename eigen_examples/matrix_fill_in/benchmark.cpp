#include <iostream>
#include <Eigen/Dense>
#include <ctime>

using namespace Eigen;
using namespace std;

int main() {
    const int I = 100;
    const int J = 100;
    const int K = 100;
    VectorXd linspace = VectorXd::LinSpaced(100, 0, 99);
    // Reshape the vector into a 10x10 matrix
    MatrixXd fill = Map<MatrixXd>(linspace.data(), 10, 10);
    float norm = 0.0;

    clock_t t0 = clock();
    for (int iter = 0; iter < 100; ++iter) {
        MatrixXd mtx = MatrixXd::Zero(I, J);
        for (int i = 0; i < I / 10; ++i) {
            for (int j = 0; j < J / 10; ++j) {
                for (int k = 0; k < K; ++k) {
                    if ((i + j) % 2 == 0) {
                        mtx.block(10 * i, 10 * j, 10, 10) += k * (fill * fill);
                    } else {
                        mtx.block(10 * i, 10 * j, 10, 10).array() += 1;
                    }
                }
            }
        }
        norm = mtx.norm();
    }
    cout << "dt = " << double(clock() - t0) / CLOCKS_PER_SEC << endl;
    cout << norm << endl;
    return 0;
}

