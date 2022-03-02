#ifndef Matrix_hpp
#define Matrix_hpp

#include <iostream>
#include <vector>
#include <ostream>
#include <fstream>
#include <cmath>
#include "LoggerAndExeption.h"
#include <cassert>
//template<typename T>
//class FileLogger;
enum class Mode {
    bin,
    txt
};


template<typename T> class Matrix;  // pre-declare the template class itself


template <typename T>
Matrix<T> AdamarsMultiplication (Matrix<T> & , Matrix<T> & );

template <typename T>
Matrix<T> transp (Matrix<T> & );

template <typename T>
Matrix<T> concateMatrix (Matrix<T> & , Matrix<T> & );

template <class T>
Matrix<T> swapRows (Matrix<T>  ,int index1, int index2);

template <class T>
Matrix<T> swapColumns (Matrix<T> , int index1, int index2);

template <class T>
Matrix<T> eraseColumns (Matrix<T> ,int index1, int index2);

template <class T>
Matrix<T> eraseRows (Matrix<T>, int index1, int index2);

template <class T>
T det (Matrix<T>);


template <class T>
int rank(Matrix<T>);


template <class T>
T trace (Matrix<T>);

template <class T>
T ScalarMultiplication( Matrix<T>,Matrix<T>);

template <class T>
double AngleBetweenVectors( Matrix<T>,Matrix<T>);

template<typename T> std::ostream& operator<< (std::ostream& out, const Matrix<T> &matrix);
template<typename T>  std::ofstream& operator<<(std::ofstream& out, Matrix<T> const & matrix);
template<typename T> std::istream& operator>>(std::istream& in, Matrix<T>& matrix);
template<typename T> Matrix<T> operator + (Matrix<T>& matrix1, Matrix<T>& matrix2);
template<typename T> Matrix<T> operator - (Matrix<T>& matrix1, Matrix<T>& matrix2);
template<typename T> Matrix<T> operator * (Matrix<T>& matrix1, Matrix<T>& matrix2);
template<typename T> Matrix<T> operator * (Matrix<T>& matrix1, T number);
//template<typename T> std::ofstream& operator>>(std::ofstream& in, Matrix<T>& matrix);

using  std::swap;





typedef std::pair<std::string, Matrix<double>> pair;
//typedef std::pair<std::string, Matrix<int>> pair;
template<class T>
class Matrix{
protected:
    int n=0, m=0;
    std::vector<std::vector<T>> array;
public:
    
    enum rowcol{
        HORIZONTAL,
        VERTICAL
    };
    Matrix ():n(0),m(0){}
    typedef std::pair<std::string, Matrix<T>> pair;
    Matrix (int N, int M){
        
        n = N;
        m = M;
        std::generate_n(std::back_inserter(array), N, [M]() -> std::vector<T>{
            std::vector<T> ivec;
            std::generate_n(std::back_inserter(ivec), M, []() { return 0; });
                return ivec;
            });
//        nullrow.resize(m);
//        std::fill(nullrow.begin(), nullrow.end(), 0);
//        nullcol.resize(n);
//        std::fill(nullcol.begin(), nullcol.end(), 0);
    }
    Matrix (int N, int M, std::vector<T> data){
        
        n = N;
        m = M;
        if(data.size()!=m*n){
            std::cerr<<"Vec lenght is not equal to correct size\n matrix may be set incorrectly\n";
        }
        int i=0;
        std::generate_n(std::back_inserter(array), N, [M,&i, data]() -> std::vector<T>{
            std::vector<T> ivec;
            std::generate_n(std::back_inserter(ivec), M, [&i, data]()mutable {return data[i++]; });
            return ivec;
        });
    }
    Matrix (int N, int M, std::vector<std::vector<T>> data){
        
        n = N;
        m = M;
        if(data.size()!=N||data[0].size()!=M){
            std::cout<<data.size()<<"\n"<<data[0].size();
            std::cerr<<"Vec lenght is not equal to correct size\n matrix may be set incorrectly\n";
        }
        
        
        std::vector<T> tmpVec;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                tmpVec.push_back(data[i][j]);
            }
        }
        new (this) Matrix(N, M, tmpVec);

//            int i=0;
//            std::generate_n(std::back_inserter(array), N, [M,&i, data]() -> std::vector<T>{
//                std::vector<T> ivec;
//                std::generate_n(std::back_inserter(ivec), M, [&i, data]()mutable {return data[i++]; });
//                return ivec;
//            });
            
    
    }
    
    Matrix (int N,std::vector<T> data,  rowcol rowOrCol){
        
        if (rowOrCol==HORIZONTAL){
            new (this) Matrix(1, N, data);
       }
        
       else if(rowOrCol==VERTICAL){
       
           new (this) Matrix(N ,1, data);
           
       }
        
    }
    Matrix (int N,std::vector<std::vector<T>> data, rowcol rowOrCol){
        
        if (rowOrCol==HORIZONTAL){
            std::vector<T> tmpVec;
            for(int i=0;i<data.size();i++){
                for(int j=0;j<data[0].size();j++){
                    tmpVec.push_back(data[i][j]);
                }
            }
            new (this) Matrix(1, N, tmpVec);
       }
        
       else if(rowOrCol==VERTICAL){
           std::vector<T> tmpVec;
           for(int i=0;i<data.size();i++){
               for(int j=0;j<data[0].size();j++){
                   tmpVec.push_back(data[i][j]);
               }
           }
           new (this) Matrix(N ,1, tmpVec);
           
       }
        
    }
    Matrix ( int N, rowcol rowOrCol){
        
        if (rowOrCol==HORIZONTAL){
            
            n = 1;
            m = N;
            std::generate_n(std::back_inserter(array), N, [N]() -> std::vector<T>{
               std::vector<T> ivec;
               std::generate_n(std::back_inserter(ivec), N, []() { return 0; });
                   return ivec;
               });
            
       }
       else if(rowOrCol==VERTICAL){
          n = N;
          m = 1;
           std::generate_n(std::back_inserter(array), N, [N]() -> std::vector<T>{
              std::vector<T> ivec;
              std::generate_n(std::back_inserter(ivec), N, []() { return 0; });
                  return ivec;
           });
           
       }
        
    }

    int GetHeight (){
      return n;
        
    }
    int GetLenght (){
        return m;
        
    }
   
    
//
    
    
    
    
    
    
    friend std::ostream& operator<<<>(std::ostream& out, const Matrix<T> &matrix);
    friend std::ofstream& operator<<<>(std::ofstream& out, Matrix<T> const& matrix);
    friend std::istream& operator>><>(std::istream& in, Matrix<T>& matrix);
    friend Matrix<T> operator + <>(Matrix<T>& matrix1, Matrix<T>& matrix2);
    friend Matrix<T> operator - <>(Matrix<T>& matrix1, Matrix<T>& matrix2);
    friend Matrix<T> operator * <>(Matrix<T>& matrix1, Matrix<T>& matrix2);
    friend Matrix<T> operator * <>(Matrix<T>& matrix1, T number);
    
//    friend std::ofstream& operator>><>(std::ofstream& in, Matrix<T>& matrix);
//    friend std::ostream& operator<< (std::ostream &out, const Matrix &matrix);
    
    std::vector<T>& operator [] (int index){
        if(index<0||index>=n){
            std::string errorStr="Index of row you want to get is out of range\n";
            std::cerr<<errorStr;
            
            insertIntoFuncLogFile<T>(*this, FileLogger<T>::e_logType::LOG_ERROR, __LINE__, __func__, errorStr);
         
        }
        else{
            if(n!=0){
                return array[index];
            }
            else{
                std::string errorStr="Matrix is empty!!!\nEnd of program\n";
                std::cerr<<errorStr;
                insertIntoFuncLogFile( *this, FileLogger<T>::e_logType::LOG_ERROR,__LINE__, __func__,errorStr );
                exit(EXIT_FAILURE);
            }
        }
        return array[index];
    }
    Matrix resize(int height, int lenght){
        if(n*m!=height*lenght){
            std::cerr<<"You cant resize "<<n<<" "<<m<<" to "<<height<<"  "<<lenght<<"\n";
            std::cerr<<"operator will throw pair<Exeption,Zero matrix>\n";
            throw pair("Resize method is impossible You cant resize "+n+" "+m+" to "+height+"  "+lenght+"\n", *this);
            return *this;
        }
        
        std::vector<T> tmp;
        for(int i = 0;i < n;i++){
            for(int j = 0;j < m;j++){
                tmp.push_back(array[i][j]);
            }
        }
        *this=Matrix<T>(height, lenght, tmp);
        return *this;
    }
    
    Matrix& AdamarsMultiplication ( Matrix<T> & matrix){
        
        Matrix<int> resultMatrix = Matrix<int> (n, matrix.GetLenght());
        if(n!=matrix.GetHeight()|| m!=matrix.GetLenght()){
            std::string errorStr="Matrix shapes are not equal, Adamars multiplication is impossible\n";
            std::cerr<<errorStr;
            throw pair("Matrix shapes are not equal, Adamars multiplication is impossible\n method will throw pair<Exeption, *this matrix>\n",*this);
            
        }
        for (int i = 0; i < n; ++ i){
            for (int j = 0; j < n; ++ j){
                resultMatrix[i][j]=array[i][j]*matrix[i][j];
            }
        }
        *this=resultMatrix;
        return *this;
    }
    
    
    //НЕ РАБОТАЕТ
    T det() {//Определитель
        if(n!=m){
            std::string errorStr="Matrix shapes are not equal, Det  is impossible\n";
            std::cerr<<errorStr;
            throw pair("Matrix shapes are not equal, Det  is impossible\n method will throw pair<Exeption, *this matrix>\n",*this);
            
        }
        Matrix<T> tmp(n,m,array);
        long int n = array.size();
        double res = 1;


           for(int col = 0; col < n; ++col) {
              bool found = false;
              for(int row = col; row < n; ++row) {
                 if(tmp[row][col]) {
                    if ( row != col )
                    {
                        tmp[row].swap(tmp[col]);
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
                    int del = tmp[row][col] / tmp[col][col];
                    for (int j = col; j < n; ++j) {
                        tmp[row][j] -= del * tmp[col][j];
                    }
                    if (tmp[row][col] == 0)
                    {
                       break;
                    }
                    else
                    {
                        res*=-1;
                        tmp[row].swap(tmp[col]);
                    }
                 }
              }
           }
           for(int i = 0; i < n; ++i) {
              res *= tmp[i][i];
           }
           return res;
        }

    Matrix& transp (){
        std::vector < std::vector <double> >tmpvec (m, std::vector <double> (n) );
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                tmpvec[j][i]=array[i][j];
            }
        }
        Matrix tmp (m, n, tmpvec);
        *this=tmp;

        return *this;
    }
    
//    // транспонирование матрицы
//    Matrix<T>& transp (){
//       Matrix<T> tmp (m, n, array);
//        *this=tmp;
//        for (int i = 0; i < n; ++ i){
//            for (int j = 0; j < i; ++ j){
//                swap(array[i][j], array[j][i]);
//            }
//        }
//
//        return *this;
//    }

    std::vector<std::vector<T>> getArray(){
        return array;
    }
   // получить строку матрицы
    Matrix<T> getRow (int index){
//        assert((index >= 0 && index < n)&&"Index of row you want to get out of range");
        if(index <= 0 || index > n){
            std::string errorStr="Index of row you want to get out of range\n";
            
            throw pair(errorStr+"method GetRow() will throw pair<exeption, zero Row>", Matrix<T>(this->array[0],this->M, HORIZONTAL, true));
        }
        Matrix<T> row(array[index],m,HORIZONTAL);
        return row;
    }
    
   // получить столбец матрицы
    
    Matrix<T> getColumn (int index){
        std::vector<T> c;
        if(index<0||index>=m){
            for (int i = 0; i < n; ++ i){
                c.push_back(0);
            }
            Matrix col(c,n,VERTICAL);
            std::string errorStr="Index of col you want to get out of range";
            throw pair(errorStr+"method getColumn() will throw pair<exeption, zero Column>", col);
        }
        
        for (int i = 0; i < n; ++ i){
            c.push_back(this->a[i][index]);
        }
        Matrix col(c,n,VERTICAL);
        return col;
    }
    
   // поменять местами две строки
    Matrix& swapRows (int index1, int index2){
        if (index1 < 0 || index2 < 0 || index1 >= n || index2 >= n){
            
            std::string errorStr="index of swaping rows is out of range\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method swapRows() will throw pair<exeption, this matrix>", *this);
        }
        for (int i = 0; i < m; ++ i){
            swap (array[index1][i], array[index2][i]);
        }
        return *this;
    }
   // поменять местами два столбца
    Matrix& swapColumns (int index1, int index2){
        if (index1 < 0 || index2 < 0 || index1 >= m || index2 >= m){
            std::string errorStr="index of swaping columns is out of range\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method swapColumns() will throw pair<exeption, this matrix>", *this);
        }
          
        for (int i = 0; i < n; ++ i){
            swap (array[i][index1], array[i][index2]);
        }
        return *this;
         
    }
    Matrix& concateMatrix (Matrix & matrix){
        if (n != matrix.GetHeight()){
            std::string errorStr="Height of matrix you want to concatenate is not equal to first matrix height\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method concateMatrix() will throw pair<exeption, this matrix>", *this);
        }
        Matrix resultMatrix = Matrix (n, m + matrix.GetLenght());
        for (int i = 0; i < n; ++ i){
            for (int j = 0; j < m; ++ j)
                resultMatrix[i][j] = array[i][j];
        for (int j = 0; j < matrix.GetLenght(); ++ j)
            resultMatrix[i][j + m] = matrix[i][j];
            
        }
        *this=resultMatrix;
        return *this;
    }
   // удаление столбцов с index1 по index2
    Matrix& eraseColumns (int index1, int index2){
        if(index1>=m||index2>=m){
            std::string errorStr="indexes of columns you want to errase are out of range\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method eraseColumns() will throw pair<exeption, this matrix>", *this);
        }
        Matrix resultMatrix = Matrix (n, m - (index2 - index1 + 1));
        for (int j = 0; j < index1; ++ j){
            for (int i = 0; i < n; ++ i){
                resultMatrix[i][j] = array[i][j];
            }
        }
        for (int j = index2 + 1; j < m; ++ j){
            for (int i = 0; i < n; ++ i){
                resultMatrix[i][j - index2 - 1 + index1] = array[i][j];
            }
        }
        *this=resultMatrix;
        return *this;
    }
    Matrix& eraseRows (int index1, int index2){
        if(index1>=n||index2>=n){
            std::string errorStr="indexes of columns you want to errase are out of range\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method eraseRows() will throw pair<exeption, this matrix>", *this);
        }
        Matrix resultMatrix = Matrix (n- (index2 - index1 + 1), m );
        for (int i = 0; i < index1; ++ i){
            for (int j = 0; j < m; ++ j){
                resultMatrix[i][j] = array[i][j];
            }
        }
        for (int i = index2 + 1; i < m; ++ i){
            for (int j = 0; j < n; ++ j){
                resultMatrix[i][j - index2 - 1 + index1] = array[i][j];
            }
        }
        *this=resultMatrix;
        return *this;
    }
    int rank(){
        const double EPS = 1E-9;
        int rank = (n>=m)?n:m;
        Matrix<T> tmp(n,m,array);
        std::vector<char> line_used (n);
        for (int i=0; i<m; ++i) {
            int j;
            for (j=0; j<n; ++j){
                if (!line_used[j] && abs(tmp[j][i]) > EPS)
                    break;
            }
            if (j == n){
                --rank;
            }
            else {
                line_used[j] = true;
                for (int p=i+1; p<m; ++p){
                    tmp[j][p] /= tmp[j][i];
                }
                for (int k=0; k<n; ++k){
                    if (k != j && abs (tmp[k][i]) > EPS){
                        for (int p=i+1; p<m; ++p){
                            tmp[k][p] -= tmp[j][p] * tmp[k][i];
                        }
                    }
                }
            }
        }
        return rank;
    }
    
    T trace(){
        if(n!=m){
            std::string errorStr="matrix is not squared, trace is impossible\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method trace() will throw pair<exeption, this matrix>", *this);
        }
        T trace=0;
        for (int i=0; i<m; ++i) {
            trace+=array[i][i];
        }

        return trace;
    }
    T ScalarMultiplication( Matrix<T> vector2){
        T result = 0;
        int n1=this->GetHeight();
        int n2=vector2.GetHeight();
        int m1=this->GetLenght();
        int m2=vector2.GetLenght();
        
        if((n1!=1&&m1!=1)||(n2!=1&&m2!=1)){
            
            std::string errorStr="vectors size are not equal, Scalar Multiplication is impossible\n";
            std::cerr<<errorStr;
            throw pair(errorStr+"method ScalarMultiplication() will throw pair<exeption, this matrix>", *this);
        }
        int Max=(n1>=m1)?n1:m1;
        
        Matrix<T> tmp1(Max, array, HORIZONTAL);
        Matrix<T> tmp2(Max, vector2.array, HORIZONTAL);
        
        for(int i=0;i<Max;i++){
            result+= tmp1[0][i]*tmp2[0][i];
        }
        
        return result;
    }
    double FrobenNorm(){
        double sum=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                sum+=array[i][j]*array[i][j];
            }
        }
        return pow(sum, 0.5);
    }
    
    double EuclideanNorm(){
        double euclideanNorm = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                euclideanNorm += array[i][j]*array[i][j];
            }
        }
        euclideanNorm = std::sqrt(euclideanNorm);
        return euclideanNorm;
    }
    
    T MaxNorm(){
        T maxNorm = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if(fabs(array[i][j])>maxNorm){
                    maxNorm = fabs(array[i][j]);
                }
            }
        }
        return maxNorm;
    }
    
//    Matrix<double> Inverse()
//    {
//        if(n!=m){
//            std::string errorStr="Not squared matrix, can't find its inverse";
//            std::cerr<<errorStr;
//            throw pair(errorStr+"method Inverse() will throw pair<exeption, this matrix>", *this);
//        }
//        Matrix<double> result(n,n);
//        // Find determinant of A[][]
//        int det = this->det();
//        if (det == 0)
//        {
//
//            std::string errorStr="Singular matrix, can't find its inverse";
//            std::cerr<<errorStr;
//            throw pair(errorStr+"method Inverse() will throw pair<exeption, this matrix>", *this);
//        }
//        Matrix<T> adj(n,m);
//
//        adjoint(*this, adj);
//
//            // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
//        for (int i=0; i<n; i++)
//            for (int j=0; j<n; j++)
//                result[i][j] = adj[i][j]/float(det);
//
//        return result;
//    }
    Matrix<double>Inverse(){
        double temp;
        if(n!=m){
            std::string errorStr="Not squared matrix, can't find its inverse";
            std::cerr<<errorStr;
            throw pair(errorStr+"method Inverse() will throw pair<exeption, this matrix>", *this);
        }
        Matrix<double> result(n,n);
        std::vector<std::vector<double>>copy(n, std::vector <double> (m));
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                copy[i][j]=(array[i][j]);
            }
        }
        Matrix<double> tmp(n,n, copy);

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
        
        return result;
    }
    double AngleBetweenVectors( Matrix<T> vector2){
        if(n!=1&&m!=1){
            
        }
        double angle;
        angle = ScalarMultiplication(vector2)/(this->EuclideanNorm() *vector2.EuclideanNorm());
        angle = acos(angle);
        return angle;
    }
//    bool LoadIntoFile(const std::string& directory, Mode m) const {
//        std::ofstream output_stream;
//        if (m == Mode::bin) {
//            output_stream.open(directory, std::ifstream::binary);
//            if (output_stream.is_open()) {
//                auto[rows, columns] = GetSize();
//                output_stream.write((char*)&rows, sizeof rows);
//                output_stream.write((char*)&columns, sizeof columns);
//                for (int i = 0; i < rows; i++)
//                    for (int j = 0; j < columns; j++)
//                        output_stream.write((char*)&((*this)[i][j]), sizeof(double));
//            }
//            else return false;
//        }
//        else if (m == Mode::txt) {
//            output_stream.open(directory);
//            if (output_stream.is_open()) {
//                output_stream << *this;
//            }
//            else return false;
//        }
//        return true;
//    }
   
    void fromBinary(const std::string &name) {
        std::ifstream rf(name, std::ios::in | std::ios::binary);
        if(!rf) {
            std::string error_str="file is not opened\n";
            std::cerr<<error_str;
            throw error_str;
        }

        rf.read((char*)&n, sizeof(size_t));
        rf.read((char*)&m, sizeof(size_t));
        array.resize(n, std::vector<T>(m));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                rf.read((char*)&array[i][j], sizeof(T));
            }
        }

        rf.close();

        if(!rf.good()) {
            throw ("Error occurred during reading from file");
        }

    }
//    void toBinary(const std::string &name) const {
//        std::ofstream wf(name, std::ios::out | std::ios::binary);
//        if(!wf) {
//            throw("Cannot open the file : " + name);
//        }
//
//        wf.write((char*)&n, sizeof(size_t));
//        wf.write((char*)&m, sizeof(size_t));
//        for (size_t i = 0; i < n; ++i) {
//            for (size_t j = 0; j < m; ++j) {
//                wf.write((char*)*array[i][j], sizeof(T));
//            }
//        }
//
//        wf.close();
//
//        if(!wf.good()) {
//            throw ("Error occurred during writing to file");
//        }
//    }
    void write_bin(std::ofstream& out) {
        if(!out.is_open()) {
            std::cerr<<"File not found"<<std::endl;
            exit(-1);
        }
    
        double num;
        out.write((char*)(&n), sizeof(int));
        out.write((char*)(&m), sizeof(int));
        for(int i=0; i<n; i++) {
            for(int j= 0; j<m; j++) {
                num= array[i][j];
                out.write((char*)(&num), sizeof(double));
            }
        }
    }
//    void get_bin(std::ifstream& in) {
//        if(!in.is_open()) {
//            std::cerr<<"File not found"<<std::endl;
//            exit(-1);
//        }
//        double num;
//
//        in.read((char*)(&n), sizeof(int));
//        in.read((char*)(&m), sizeof(int));
//        for(int i=0; i<n; i++) {
//            array->push_back(new std::vector<double>);
//            for(int j= 0; j<m; j++) {
//                if(!in.read((char*)(&num), sizeof(double))) {
//                    std::cerr<<"The file is damaged"<<std::endl;
//                    exit(-1);
//                }
//                array->at(i)->push_back(num);
//            }
//        }
//    }

//
//    bool Matrix::ReadFromFile(const string& directory, Mode m) {
//        ifstream input_stream;
//        if (m == Mode::bin) {
//            input_stream.open(directory, std::ifstream::binary);
//            if (input_stream.is_open()) {
//                input_stream.read((char*)&rows_count, sizeof(int));
//                input_stream.read((char*)&columns_count, sizeof(int));
//                Reset(rows_count, columns_count);
//                for (int i = 0; i < rows_count; i++)
//                    for (int j = 0; j < columns_count; j++)
//                        input_stream.read((char*)&At(i, j), sizeof(double));
//            }
//            else return false;
//        }
//        else if (m == Mode::txt) {
//            input_stream.open(directory);
//            if (input_stream.is_open()) {
//                input_stream >> *this;
//            }
//            else return false;
//        }
//        return true;
//    }
    std::pair<int, int> GetSize() const {
        return std::make_pair(n, m);
    }



};

template <typename T>
class IdentityMatrix:public Matrix<T>{

public:
    IdentityMatrix(int n):Matrix<T>(n,n){
        for(int i=0;i<n;i++){
            Matrix<T>::array[i][i]=1;
        }
    }
//    void write_bin(std::ofstream& out) {
//        if(!out.is_open()) {
//            std::string error_str="File not found\n";
//            std::cerr<<error_str;
//            throw(error_str);
//        }
//        if(this->n==0||this->m==0) {
//            std::string error_str="Memory Error\nMatrix not specified\n";
//            std::cerr<<error_str;
//            throw(error_str);
//        }
//        double num;
//        out.write((char*)(&this->m), sizeof(int));
//        out.write((char*)(&this->n), sizeof(int));
//        for(int i=0; i<this->m; i++) {
//            for(int j= 0; j<this->n; j++) {
//                num= get(i, j);
//                out.write((char*)(&num), sizeof(double));
//            }
//        }
//    }
//
    
};
template <typename T>
class DiagonalMatrix:public Matrix<T>{
public:
    DiagonalMatrix(int n):Matrix<T>(n,n){}
    DiagonalMatrix(int n, std::vector<T> vec):Matrix<T>(n,n){
        if(vec.size()!=n){
            std::string errorStr="Lenght of vector is bigger than matrix diagonal\n";
            std::cerr<<errorStr;
            insertIntoFuncLogFile(Matrix<T>::array, FileLogger<T>::e_logType::LOG_WARNING,__LINE__, __func__,errorStr );
        }
        for(int i=0;i<n;i++){
            Matrix<T>::array[i][i]=vec[i];
        }
    }
};

template <typename T>
class UpperTriangleMatrix:public Matrix<T>{
public:
    UpperTriangleMatrix(int n):Matrix<T>(n,n){}
    UpperTriangleMatrix(int n, std::vector<T> vec):Matrix<T>(n,n){
        int count=0;
        for(int i=1;i<n;i++){
            count+=i;
        }
        if(vec.size()!=count){
            std::cerr<<"Vec lenght is not equal to correct size\n matrix may be set incorrectly\n";
        }
        count=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(j>=i){
                    Matrix<T>::array[i][j]=vec[count];
                    count++;
                }
            }
        }
    }
    
};
template <typename T>
class LowerTriangleMatrix:public Matrix<T>{
public:
    LowerTriangleMatrix(int n):Matrix<T>(n,n){}
    LowerTriangleMatrix(int n, std::vector<T> vec):Matrix<T>(n,n){
        int count=0;
        for(int i=1;i<n;i++){
            count+=i;
        }
        if(vec.size()!=count){
            std::cerr<<"Vec lenght is not equal to correct size\n matrix may be set incorrectly\n";
        }
        count=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(i>=j){
                    Matrix<T>::array[i][j]=vec[count];
                    count++;
                }
            }
        }
    }
    
};
template <typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T> & matrix){
    out<<std::endl;
     for (int i = 0; i < matrix.n; ++ i){
         for (int j = 0; j < matrix.m; ++ j){
             out<<matrix.array[i][j]<<" ";
         }
         out<<std::endl;
     }

    return out;

}
template <typename T>
std::ofstream& operator<<(std::ofstream& out, Matrix<T> const& matrix){
     for (int i = 0; i < matrix.n; ++ i){
         for (int j = 0; j < matrix.m; ++ j){
             out<<matrix.array[i][j]<<" ";
         }
         out<<std::endl;
     }
    out<<std::endl;
    return out;
}


//template <typename T>
//std::ostream & operator <<(std::ostream & output, Matrix<T> const& matrix){
//    output << "Test\n";
//
//    return output;
//}

template <typename T>
std::istream& operator>>(std::istream& in, Matrix<T>& matrix){
    int n=matrix.GetHeight(), m=matrix.GetLenght();
    //
    //std::cin>>n>>m;
    matrix=Matrix<T>(n,m);
    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < m; ++ j){
            //std::cout<<i<<" "<<j<<"\n";
            in>> matrix[i][j];
        }
    return in;
}

template <typename T>
Matrix<T> operator + (Matrix<T>& matrix1, Matrix<T>& matrix2){
    Matrix<T> resultMatrix = Matrix<T> (matrix1.GetHeight(), matrix1.GetLenght());
    if( matrix1.GetHeight() != matrix2.GetHeight() || matrix1.GetLenght() != matrix2.GetLenght()){
        std::string errorStr="Matrix shapes are not equal, operator + is impossible\n";
        std::cerr<<errorStr;
     
        throw pair(errorStr+"operator will throw pair<Exeption,Zero matrix>\n", resultMatrix);

    }
    for (int i = 0; i < matrix1.GetHeight(); ++ i){
        for (int j = 0; j < matrix1.GetLenght(); ++ j){
            resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return resultMatrix;
}
template <typename T>
Matrix<T> operator - (Matrix<T>& matrix1, Matrix<T>& matrix2){
    Matrix<T> resultMatrix = Matrix<T> (matrix1.GetHeight(), matrix1.GetLenght);
    if(matrix1.GetHeight() != matrix2.GetHeight() && matrix1.GetLenght() != matrix2.GetLenght()){
        std::string errorStr="Matrix shapes are not equal, operator - is impossible\n";
        std::cerr<<errorStr;
        throw pair(errorStr+"operator will throw pair<Exeption,Zero matrix>\n", resultMatrix);
    }

    for (int i = 0; i < matrix1.GetHeight(); ++ i)
        for (int j = 0; j < matrix1.GetLenght(); ++ j)
            resultMatrix[i][j] = matrix1[i][j] - matrix2[i][j];
    return resultMatrix;
}
template <typename T>
Matrix<T> operator * (Matrix<T>& matrix1, Matrix<T>& matrix2){
    Matrix<T> resultMatrix = Matrix<T> (matrix1.GetHeight(), matrix2.GetLenght());
    if(matrix1.GetLenght() != matrix2.GetHeight() && matrix1.GetHeight() != matrix2.GetLenght()){
        std::string errorStr="Matrix shapes are not equal, operator * is impossible\n";
        std::cerr<<errorStr;
//        throw pair(errorStr+"operator will throw pair<Exeption,Zero matrix>\n", resultMatrix);
//    }
    }
    
    for (int i = 0; i < matrix1.GetHeight(); ++ i)
        for (int j = 0; j < matrix2.GetLenght(); ++ j)
            for (int k = 0; k < matrix1.GetLenght(); ++ k)
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
    return resultMatrix;
}

template <typename T>
Matrix<T> operator * (Matrix<T>& matrix1, T number){
    
    Matrix<T> resultMatrix = Matrix<T> (matrix1.GetHeight(), matrix1.GetLenght());
    for (int i = 0; i < matrix1.GetHeight(); ++ i)
        for (int j = 0; j < matrix1.GetLenght(); ++ j)
            resultMatrix[i][j] = matrix1[i][j] * number;
    return resultMatrix;
}



#endif 

