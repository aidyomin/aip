#ifndef LoggerAndExeption_h
#define LoggerAndExeption_h

#include <fstream>
#include <chrono>
#include <ctime>
#include<string>


template<typename T>
class Matrix;

template<typename T>
class FileLogger {
    
    std::ofstream myFile;
    unsigned int numWarnings;
    unsigned int numErrors;
public:
    
    enum class e_logType { LOG_ERROR, LOG_WARNING, LOG_INFO };

           
    explicit FileLogger (const char *filename="ige_log.txt" ):numWarnings (0U),numErrors (0U){
        myFile.open (filename);

        if (myFile.is_open()) {
            myFile << "Log file created" << std::endl << std::endl;
        }

    }
    


        
    ~FileLogger () {
        if (myFile.is_open()) {
            myFile << std::endl << std::endl;
            // Report number of errors and warnings
            myFile << numWarnings << " warnings" << std::endl;
            myFile << numErrors << " errors" << std::endl;
            myFile.close();
        } // if
    }


    friend FileLogger &operator << (FileLogger &logger, const e_logType l_type) {
        
        std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        switch (l_type) {
            case FileLogger::e_logType::LOG_ERROR:
                
                logger.myFile << "[ERROR]: ";
                logger.myFile << std::ctime(&time);
                ++logger.numErrors;
                break;

            case FileLogger::e_logType::LOG_WARNING:
                logger.myFile << "[WARNING]: ";
                logger.myFile << std::ctime(&time);
                ++logger.numWarnings;
                break;

            default:
                logger.myFile << "[INFO]: ";
                logger.myFile << std::ctime(&time);
                break;
        }


        return logger;
    }


    friend FileLogger &operator << (FileLogger &logger, std::string text) {
        logger.myFile << text << std::endl;
        return logger;
    }
    friend FileLogger & operator <<(FileLogger &logger, Matrix<T> matrix){
        logger.myFile<<matrix;
        return logger;
    }
    
            
    FileLogger (const FileLogger &) = delete;
    FileLogger &operator= (const FileLogger &) = delete;

};



template <typename T>
class Exception {
public:
    Exception( FileLogger<T> &logger, Matrix<T>& matrix, typename FileLogger<T>:: e_logType l_type,  int lineNumber, std::string function, std::string text): msg_(text) {
        logger<<l_type;
       
        logger<< text+ "\nline number "+std::to_string(lineNumber)+" function "+function;
//        logger<<"\n";
        logger.myFile<<matrix;
        logger<<"\n";
    }
    Exception( FileLogger<T> &logger, Matrix<T>& matrix1, Matrix<T>& matrix2,  typename FileLogger<T>:: e_logType l_type,int lineNumber, std::string function,std::string text ): msg_(text) {
        logger<<l_type;
        logger<< text+"\nline number "+std::to_string(lineNumber)+" function "+ function;;
        logger<<"Matrix1\n";
        logger<<matrix1;
        logger<<"\n";
        logger<<"Matrix2\n";
        logger<<matrix2;
        logger<<"\n";
    }
    
    ~Exception() {}
   std::string getMessage() const {return(msg_);}
    
private:
   std::string msg_;
};
template <typename T>
void insertIntoFuncLogFile(Matrix<T>& matrix, typename FileLogger<T>:: e_logType l_type,  int lineNumber, std::string function, std::string text, std::string fname="FuncrLogs"){
    std::ofstream myFile;
    myFile.open(fname, std::ios_base::app);
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    switch (l_type) {
        case FileLogger<T>::e_logType::LOG_ERROR:
            
            myFile << "[ERROR]: ";
            myFile << std::ctime(&time);
            break;

        case FileLogger<T>::e_logType::LOG_WARNING:
            myFile << "[WARNING]: ";
            myFile << std::ctime(&time);
            break;

        default:
            myFile << "[INFO]: ";
            myFile << std::ctime(&time);
            break;
    } // s
    myFile<< text+"\nline number "+std::to_string(lineNumber)+" function "+ function+"\n";
    myFile<< matrix;
    myFile<< "\n";
}



template <typename T>
void insertIntoFuncLogFile(Matrix<T>& matrix1, Matrix<T>& matrix2, typename FileLogger<T>:: e_logType l_type,  int lineNumber, std::string function, std::string text, std::string fname="FuncrLogs.txt"){
    std::ofstream myFile;
    myFile.open(fname, std::ios_base::app);
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    switch (l_type) {
        case FileLogger<T>::e_logType::LOG_ERROR:
            
            myFile << "[ERROR]: ";
            myFile << std::ctime(&time);
            break;

        case FileLogger<T>::e_logType::LOG_WARNING:
            myFile << "[WARNING]: ";
            myFile << std::ctime(&time);
            break;

        default:
            myFile << "[INFO]: ";
            myFile << std::ctime(&time);
            break;
    } // s
    myFile<< text+"\nline number "+std::to_string(lineNumber)+" function "+ function+"\n";;
    
    myFile<<"Matrix1\n";
    myFile<<matrix1;
    myFile<<"\n";
    myFile<<"Matrix2\n";
    myFile<<matrix2;
    myFile<<"\n";
}

#endif
