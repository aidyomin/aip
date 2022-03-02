#include "Matrix.hpp"
#include "LoggerAndExeption.h"

#define T double


int main(){

    FileLogger<double> logger;
//    Matrix<T> matrix1;
//    Matrix<T> matrix2;
//    std::cin>>matrix2;
//    std::cin>>matrix1;
//    std::vector<T> v={1,2,3,4,5,6,7,7,2,2,345,23,45,2346,3246,46,47,};
//
//    std::vector<T> v2={4,3,2};
//
//    Matrix<T> matrix3(3,5,v);
//    Matrix<T> matrix4(3,v2,Matrix<int>::HORIZONTAL);
//    std::cout<<"\n";
//    std::cout<<matrix3;
//    std::cout<<"\n";
//    std::cout<<matrix4;
//    std::cout<<"\n";
//    std::cout<<matrix3.AngleBetweenVectors(matrix4);
//    std::cout<<"\n";
//    std::cout<<AngleBetweenVectors(matrix3, matrix4);
    std::cout<<"\n";
    std::ifstream readFromFile;
    readFromFile.open("scores.txt",std::ios_base::in);
    
 
   
    Matrix<double> matrix1 (2,2);
    readFromFile>>matrix1;
    std::cout<<matrix1<<"\n";
    std::cout<<matrix1.det();
 
    std::vector<double> v2={1,2,3,4,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,6,7,8,9,1,2,3,4,5,6,7,8,9};
    Matrix<double> matrix2(5,5,v2);
    
    std::cout<<matrix2<<"\n";
    std::cout<<matrix2.det()<<"\n";
    std::cout<<matrix1<<"\n";
    std::cout<<matrix1.det()<<"\n";
//    std::cout<<matrix1;
//    std::cout<<matrix1;

    std::ofstream fout("output.txt");
    fout<< matrix1;
    
   // std::cout<<AdamarsMultiplication( matrix1, matrix2)<<"\n";
//    std::cout<<"\n";
//    std::cout<<AdamarsMultiplication<T>( matrix2, matrix1)<<"\n";

 
//    std::cin>>matrix2;
//
//    std::ofstream bin_fout("bin_output.txt", std::ios::binary);
//    matrix1.write_bin(bin_fout);
//    bin_fout.close();

//    std::ifstream bin_in("bin_output.bin");
//    Matrix<T> matrix4(3,3);
//    matrix4.
    //std::ifstream bin_in("bin_output.bin");
//    Matrix<T> matrix4(3,3);
//    matrix4.fromBinary("bin_output.bin");
//    //bin_in.close();
//    std::cout<<matrix4;
    
    Matrix<double> matrix3;
    try{
        matrix3=matrix1+matrix2;
        std::cout<<matrix3;
    }
    catch(std::pair<std::string, Matrix<double>>pair){
        Exception<double> exeption {logger, matrix1, matrix2, FileLogger<double>::e_logType::LOG_ERROR,__LINE__, __func__, pair.first};
        matrix3=pair.second;
        std::cout<<matrix3;
    }


    return 0;
}
