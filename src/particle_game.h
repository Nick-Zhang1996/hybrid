#pragma once

typedef Eigen::MatrixXd np_array;

class ParticleGame {
    private:
        np_array A,B;

    public:
        ParticleGame(){}
        // TODO unnecessary copy
        void set_A(const Eigen::MatrixXd &val){
            A = np_array(val);
        }
        void set_B(const Eigen::MatrixXd &val){
            B = np_array(val);
        }
        np_array f(const np_array x, const np_array u){
            return A * x + B * u;
        }

};
