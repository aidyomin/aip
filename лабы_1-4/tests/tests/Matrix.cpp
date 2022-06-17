#include "Matrix.hpp"
#include "LoggerAndExeption.h"

template <typename T>
Matrix<T> AdamarsMultiplication (Matrix<T> & matrix1, Matrix<T> & matrix2){
    Matrix<T> resultMatrix = Matrix<T> (matrix1.GetHeight(), matrix1.GetLenght());
    if(matrix1.GetHeight()!=matrix2.GetHeight()|| matrix1.GetLenght()!=matrix2.GetLenght()){
        std::string errorStr="Matrix shapes are not equal, Adamars multiplication is impossible\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix1, matrix2, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return resultMatrix;
    }
    
    for (int i = 0; i < matrix1.GetHeight(); ++ i){
        for (int j = 0; j < matrix1.GetLenght(); ++ j){
            resultMatrix[i][j]=matrix1[i][j]*matrix2[i][j];
        }
    }
    return resultMatrix;
}

template Matrix<int> AdamarsMultiplication (Matrix<int> & , Matrix<int> & );
template Matrix<double> AdamarsMultiplication (Matrix<double> & , Matrix<double> & );




template <typename T>
Matrix<T> transp (Matrix<T> matrix){
   Matrix<T> resultMatrix = Matrix<T> (matrix.GetLenght(), matrix.GetHeight());
    for (int i = 0; i < matrix.GetHeight(); ++ i){
        for (int j = 0; j < matrix.GetLenght(); ++ j){
            resultMatrix[j][i] = matrix[i][j];
        }
    }
   return resultMatrix;
}

template Matrix<int> transp (Matrix<int> );
template Matrix<double> transp (Matrix<double> );


template <typename T>
Matrix<T> concateMatrix (Matrix<T> & matrix1, Matrix<T> & matrix2){
    if (matrix1.GetHeight()!= matrix2.GetHeight()){
        std::string errorStr="Height of matrix you want to concatenate is not equal to first matrix height\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix1, matrix2, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return matrix1;
    }
    Matrix resultMatrix = Matrix<T> (matrix1.GetHeight(), matrix1.GetLenght() + matrix2.GetLenght());
    for (int i = 0; i < matrix1.GetHeight(); ++ i){
        for (int j = 0; j < matrix1.GetLenght() ; ++ j){
            resultMatrix[i][j] = matrix1[i][j];
        }
        for (int j = 0; j < matrix2.GetLenght(); ++ j){
        resultMatrix[i][j + matrix1.GetLenght()] = matrix2[i][j];
        }
    }
    return resultMatrix;
}



template Matrix<int> concateMatrix (Matrix<int> &, Matrix<int> &);
template Matrix<double> concateMatrix (Matrix<double> &, Matrix<double> &);







template <typename T>
Matrix<T> swapRows (Matrix<T> matrix ,int index1, int index2){
    if (index1 < 0 || index2 < 0 || index1 >= matrix.GetHeight() || index2 >= matrix.GetHeight()){
        std::string errorStr="index of swaping rows is out of range\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return matrix;
    }
    for (int i = 0; i < matrix.GetLenght(); ++ i){
        swap (matrix[index1][i], matrix[index2][i]);
    }
    return matrix;
}

template Matrix<int> swapRows (Matrix<int> ,int index1, int index2);
template Matrix<double> swapRows (Matrix<double> ,int index1, int index2);

template <typename T>
Matrix<T> swapColumns (Matrix<T> matrix, int index1, int index2){
    if (index1 < 0 || index2 < 0 || index1 >= matrix.GetLenght() || index2 >= matrix.GetLenght()){
        std::string errorStr="index of swaping columns is out of range\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr);
        return matrix;
        
    }
      
    for (int i = 0; i < matrix.GetHeight(); ++ i){
        swap (matrix[i][index1], matrix[i][index2]);
    }
    return matrix;
     
}
template Matrix<int> swapColumns (Matrix<int>, int index1, int index2);
template Matrix<double> swapColumns (Matrix<double>, int index1, int index2);

template <typename T>
Matrix<T> eraseColumns (Matrix<T> matrix,int index1, int index2){
    if(index1>=matrix.GetLenght()||index2>=matrix.GetLenght()){
        std::string errorStr="indexes of columns you want to errase are out of range\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return matrix;
    }
    Matrix resultMatrix = Matrix<T> (matrix.GetHeight(), matrix.GetLenght() - (index2 - index1 + 1));
    for (int j = 0; j < index1; ++ j){
        for (int i = 0; i < matrix.GetHeight(); ++ i){
            resultMatrix[i][j] = matrix[i][j];
        }
    }
    for (int j = index2 + 1; j < matrix.GetLenght(); ++ j){
        for (int i = 0; i < matrix.GetHeight(); ++ i){
            resultMatrix[i][j - index2 - 1 + index1] = matrix[i][j];
        }
    }
    return resultMatrix;
}
template Matrix<int> eraseColumns (Matrix<int>,int index1, int index2);
template Matrix<double> eraseColumns (Matrix<double>,int index1, int index2);

template <typename T>
Matrix<T> eraseRows (Matrix<T> matrix, int index1, int index2){
    if(index1>=matrix.GetHeight()||index2>=matrix.GetHeight()){
        std::string errorStr="indexes of columns you want to errase are out of range\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return matrix;
    }
    Matrix resultMatrix = Matrix<T> (matrix.GetHeight()- (index2 - index1 + 1), matrix.GetLenght());
    for (int i = 0; i < index1; ++ i){
        for (int j = 0; j < matrix.GetLenght(); ++ j){
            resultMatrix[i][j] = matrix[i][j];
        }
    }
    for (int i = index2 + 1; i < matrix.GetLenght(); ++ i){
        for (int j = 0; j < matrix.GetHeight(); ++ j){
            resultMatrix[i][j - index2 - 1 + index1] = matrix[i][j];
        }
    }
    return resultMatrix;
}
template Matrix<int> eraseRows (Matrix<int>, int index1, int index2);
template Matrix<double> eraseRows (Matrix<double>, int index1, int index2);


template <typename T>
T det(Matrix<T> matrix) {//Определитель
    if(matrix.GetHeight()!=matrix.GetLenght()){
        std::string errorStr="Matrix shapes are not equal, Det  is impossible, func returns 0\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return 0;
    }
    long int n = matrix.GetHeight();
    int res = 1;


       for(int col = 0; col < n; ++col) {
          bool found = false;
          for(int row = col; row < n; ++row) {
             if(matrix[row][col]) {
                if ( row != col )
                {
                    matrix[row].swap(matrix[col]);
                }
                found = true;
                break;
             }
          }
          if(!found) {
             return 0;
          }
          for(int row = col + 1; row < n; ++row) {
             while(true) {
                int del = matrix[row][col] / matrix[col][col];
                for (int j = col; j < n; ++j) {
                    matrix[row][j] -= del * matrix[col][j];
                }
                if (matrix[row][col] == 0)
                {
                   break;
                }
                else
                {
                    res*=-1;
                    matrix[row].swap(matrix[col]);
                }
             }
          }
       }
       for(int i = 0; i < n; ++i) {
          res *= matrix[i][i];
       }
       return res;
    }

template int det( Matrix<int>);
template double det( Matrix<double>);

template <typename T>
int rank(Matrix<T> matrix){
    const double EPS = 1E-9;
    int n= matrix.GetHeight();
    int m=matrix.GetLenght();
    int rank = (n>=m)?n:m;
    std::vector<char> line_used (n);
    for (int i=0; i<m; ++i) {
        int j;
        for (j=0; j<n; ++j){
            if (!line_used[j] && abs(matrix[j][i]) > EPS)
                break;
        }
        if (j == n){
            --rank;
        }
        else {
            line_used[j] = true;
            for (int p=i+1; p<m; ++p){
                matrix[j][p] /= matrix[j][i];
            }
            for (int k=0; k<n; ++k){
                if (k != j && abs (matrix[k][i]) > EPS){
                    for (int p=i+1; p<m; ++p){
                        matrix[k][p] -= matrix[j][p] * matrix[k][i];
                    }
                }
            }
        }
    }
    return rank;
}
template int rank( Matrix<int>);
template int rank( Matrix<double>);

template<typename T>
T trace(Matrix<T> matrix){
    if(matrix.GetHeight()!=matrix.GetLenght()){
        std::string errorStr="matrix is not squared, trace is impossible, function return 0\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
        return 0;
    }
    T trace=0;
    T tmp;
    for (int i=0; i<matrix.GetLenght(); ++i) {
        tmp=matrix[i][i];
        trace+=tmp;
    }
    return trace;
}
template int trace( Matrix<int>);
template double trace( Matrix<double>);

template<typename T>
T ScalarMultiplication(Matrix<T> vector1, Matrix<T> vector2){
    double result = 0;
    int n1=vector1.GetHeight();
    int n2=vector2.GetHeight();
    int m1=vector1.GetLenght();
    int m2=vector2.GetLenght();
    
    if((n1!=1&&m1!=1)||(n2!=1&&m2!=1)){
        
        std::string errorStr="vectors size are not equal, Scalar Multiplication is impossible\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( vector1,vector2, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
    }
    int Max=(n1>=m1)?n1:m1;
    
    Matrix<T> tmp1(Max, vector1.getArray(), Matrix<T>::HORIZONTAL);
    Matrix<T> tmp2(Max, vector2.getArray(), Matrix<T>::HORIZONTAL);
    
    for(int i=0;i<Max;i++){
        result+= tmp1[0][i]*tmp2[0][i];
    }
    
    return result;
}
template int ScalarMultiplication( Matrix<int>,Matrix<int>);
template double ScalarMultiplication( Matrix<double>,Matrix<double>);


template<typename T>
double AngleBetweenVectors(Matrix<T> vector1, Matrix<T> vector2){
    double angle;
    angle = ScalarMultiplication(vector1, vector2)/(vector1.EuclideanNorm() *vector2.EuclideanNorm());
    angle = acos(angle);
    return angle;
}

template double AngleBetweenVectors( Matrix<int>,Matrix<int>);
template double AngleBetweenVectors( Matrix<double>,Matrix<double>);


template <typename T>
Matrix<T> reverse (Matrix<T> matrix){
    double det = matrix.det();
    if (det == 0){
        std::string errorStr="Determinant equals to zero. Matrix cannot be reversed\n";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
    }
    int n=matrix.GetHeight();
    int m= matrix.GetHeight();
    Matrix<T> big_matrix(n, m * 2);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            big_matrix[i] [j] = matrix[i] [j];
        }
    };

    for (int i = 0; i < m; ++i) {
        big_matrix[i] [m + i] = 1;
    }


    int swapsCount = 0;
    for (int i = 0; i < m && swapsCount < n; ++i) {
        double max = big_matrix[swapsCount][ i];
        int maxIndex = swapsCount;

        for (int row = swapsCount + 1; row < n; ++row) {
            if ((big_matrix[row][i] != 0) && ((max == 0) || (big_matrix[row][i] > max))) {
                max = big_matrix[row][i];
                maxIndex = row;
            }
        }

        if (max == 0) continue;

        big_matrix.swapRows(maxIndex, swapsCount++);

        for (int j = swapsCount; j < n; ++j) {
            double q = big_matrix[j][i] / big_matrix[swapsCount - 1][i];
            for (int k = 0; k < m * 2; ++k) {
                big_matrix[j][k] -= big_matrix[swapsCount - 1][k] * q;
                if (fabs(big_matrix[j][k]) < 0.00000001) big_matrix[j][k] = 0;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double general = big_matrix[i][i];
        for (int j = i; j < m * 2; ++j) {
            big_matrix[i][j] /= general;
        }

        for (int j = i; j > 0; --j) {
            double q = big_matrix[j - 1][i];
            for (int k = i; k < m * 2; ++k) {
                big_matrix[j - 1][k] -= q * big_matrix[i][k];
                if (fabs(big_matrix[j - 1][k]) < 0.00000001) big_matrix[j - 1][k] = 0;
            }
        }
    }

    Matrix<T> result(n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = m; j < m * 2; ++j) {
            result[i][j - m] = big_matrix[i][j];
        }
    }

    return result;
}
template Matrix<int> reverse (Matrix<int> );
template Matrix<double> reverse (Matrix<double> );



template <typename T>
Matrix<T>inverse(Matrix<T> matrix){
    int n=matrix.GetHeight();
    int m=matrix.GetLenght();
    double temp;
    if(n!=m){
        std::string errorStr="Not squared matrix, can't find its inverse";
        std::cerr<<errorStr;
        insertIntoFuncLogFile( matrix, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
    }
    Matrix<T> result(n,n);
    std::vector<std::vector<T>>copy(n, std::vector <T> (m));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            copy[i][j]=(matrix[i][j]);
        }
    }
    Matrix<T> tmp(n,n, copy);

    for (int i = 0; i < n; i++){
            for (int j = 0; j <n; j++){
                result[i][j] = 0.0;
                if (i == j)
                    result[i][j] = 1.0;
            }
    }

    for (int k = 0; k < n; k++){
        temp = tmp[k][k];
        for (int j = 0; j < n; j++){
            tmp[k][j] /= temp;
            result[k][j] /= temp;
        }
        for (int i = k + 1; i < n; i++){
            temp = tmp[i][k];
            for (int j = 0; j < n; j++){
                tmp[i][j] -= tmp[k][j] * temp;
                result[i][j] -= result[k][j] * temp;
            }
        }
    }

    for (int k = n - 1; k > 0; k--){
        for (int i = k - 1; i >= 0; i--){
            temp = tmp[i][k];
            for (int j = 0; j < n; j++){
                tmp[i][j] -= tmp[k][j] * temp;
                result[i][j] -= result[k][j] * temp;
            }
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j <n; j++){
            tmp[i][j] = result[i][j];
        }
    }
//    for (int i = 0; i < n; i++){
//        for (int j = 0; j <n; j++){
//            if(abs(result[i][j])<0.1){
//                result[i][j]=0;
//            }
//
//        }
//    }
    
    return result;
}
template Matrix<int> inverse (Matrix<int> );
template Matrix<double> inverse (Matrix<double> );
