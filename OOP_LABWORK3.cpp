#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <stdexcept>

// Клас TRational
class TRational {
private:
    long long numerator, denominator;

    void simplify() {
        long long g = gcd(std::abs(numerator), std::abs(denominator));
        numerator /= g;
        denominator /= g;
        if (denominator < 0) {
            numerator = -numerator;
            denominator = -denominator;
        }
    }

    long long gcd(long long a, long long b) const {
        while (b) {
            long long temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

public:
    // Конструктори
    TRational(long long num = 0, long long denom = 1) : numerator(num), denominator(denom) {
        if (denom == 0)
            throw std::invalid_argument("Denominator cannot be zero");
        simplify();
    }

    // Перетворення до double
    operator double() const {
        return static_cast<double>(numerator) / denominator;
    }

    // Оператори з TRational
    TRational operator+(const TRational& other) const {
        return TRational(numerator * other.denominator + other.numerator * denominator,
            denominator * other.denominator);
    }

    TRational operator-(const TRational& other) const {
        return TRational(numerator * other.denominator - other.numerator * denominator,
            denominator * other.denominator);
    }

    TRational operator*(const TRational& other) const {
        return TRational(numerator * other.numerator, denominator * other.denominator);
    }

    TRational operator/(const TRational& other) const {
        if (other.numerator == 0)
            throw std::invalid_argument("Division by zero");
        return TRational(numerator * other.denominator, denominator * other.numerator);
    }

    // Оператори з double
    TRational operator+(double value) const {
        return *this + TRational(static_cast<long long>(value * denominator), denominator);
    }

    TRational operator-(double value) const {
        return *this - TRational(static_cast<long long>(value * denominator), denominator);
    }

    TRational operator*(double value) const {
        return *this * TRational(static_cast<long long>(value * denominator), denominator);
    }

    TRational operator/(double value) const {
        if (value == 0.0)
            throw std::invalid_argument("Division by zero");
        return *this / TRational(static_cast<long long>(value * denominator), denominator);
    }

    // Оператори для double (зворотний порядок)
    friend TRational operator+(double value, const TRational& r) {
        return TRational(static_cast<long long>(value * r.denominator), r.denominator) + r;
    }

    friend TRational operator-(double value, const TRational& r) {
        return TRational(static_cast<long long>(value * r.denominator), r.denominator) - r;
    }

    friend TRational operator*(double value, const TRational& r) {
        return TRational(static_cast<long long>(value * r.denominator), r.denominator) * r;
    }

    friend TRational operator/(double value, const TRational& r) {
        if (r.numerator == 0)
            throw std::invalid_argument("Division by zero");
        return TRational(static_cast<long long>(value * r.denominator), r.denominator) / r;
    }

    // Перевантаження оператора виводу
    friend std::ostream& operator<<(std::ostream& os, const TRational& r) {
        os << r.numerator << "/" << r.denominator;
        return os;
    }
};


// Шаблонний клас Matrix
template <typename T>
class Matrix {
private:
    size_t rows, cols;
    std::vector<std::vector<T>> data;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<T>(c, T())) {}

    // Введення елементів
    void input() {
        std::cout << "Enter elements of the matrix (" << rows << "x" << cols << "):\n";
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cin >> data[i][j];
            }
        }
    }

    // Виведення матриці
    void print() const {
        for (const auto& row : data) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << '\n';
        }
    }

    // Перевантаження оператора доступу
    std::vector<T>& operator[](size_t index) {
        if (index >= rows) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    const std::vector<T>& operator[](size_t index) const {
        if (index >= rows) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // Додавання матриць
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices dimensions must match for addition");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] + other[i][j];
            }
        }
        return result;
    }

    // Віднімання матриць
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices dimensions must match for subtraction");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] - other[i][j];
            }
        }
        return result;
    }

    // Транспонування матриці
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }

    // Обчислення визначника (рекурсія)
    T determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square to calculate determinant");
        }

        if (rows == 1) return data[0][0];
        if (rows == 2) return data[0][0] * data[1][1] - data[0][1] * data[1][0];

        T det = 0;
        for (size_t p = 0; p < cols; ++p) {
            Matrix<T> subMatrix(rows - 1, cols - 1);
            for (size_t i = 1; i < rows; ++i) {
                size_t colIndex = 0;
                for (size_t j = 0; j < cols; ++j) {
                    if (j == p) continue;
                    subMatrix.data[i - 1][colIndex++] = data[i][j];
                }
            }
            det = det + (p % 2 == 0 ? 1 : -1) * data[0][p] * subMatrix.determinant();
        }
        return det;
    }

    // Обчислення норми
    double norm() const {
        double sum = 0;
        for (const auto& row : data) {
            for (const auto& elem : row) {
                double val = static_cast<double>(elem);
                sum += val * val;
            }
        }
        return std::sqrt(sum);
    }

    // Розв’язання системи рівнянь методом Гаусса
    std::vector<T> gaussSolve(const std::vector<T>& b) {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square to solve system of equations");
        }
        if (b.size() != rows) {
            throw std::invalid_argument("Vector size must match matrix dimensions");
        }

        // Робимо копію матриці та вектору b
        Matrix<T> mat = *this;
        std::vector<T> result = b;

        for (size_t i = 0; i < rows; ++i) {
            // Шукаємо головний елемент
            size_t maxRow = i;
            for (size_t k = i + 1; k < rows; ++k) {
                if (std::abs(static_cast<double>(mat[k][i])) > std::abs(static_cast<double>(mat[maxRow][i]))) {
                    maxRow = k;
                }
            }
            std::swap(mat[i], mat[maxRow]);
            std::swap(result[i], result[maxRow]);

            // Нормалізація головного рядка
            T divisor = mat[i][i];
            for (size_t j = i; j < cols; ++j) {
                mat[i][j] = mat[i][j] / divisor;
            }
            result[i] = result[i] / divisor;

            // Виключення стовпця
            for (size_t k = i + 1; k < rows; ++k) {
                T factor = mat[k][i];
                for (size_t j = i; j < cols; ++j) {
                    mat[k][j] = mat[k][j] - factor * mat[i][j];
                }
                result[k] = result[k] - factor * result[i];
            }
        }

        // Зворотний хід
        std::vector<T> x(rows);
        for (int i = rows - 1; i >= 0; --i) {
            x[i] = result[i];
            for (size_t j = i + 1; j < cols; ++j) {
                x[i] = x[i] - mat[i][j] * x[j];
            }
        }

        return x;
    }
};

// Клас SparseMatrix
template <typename T>
class SparseMatrix {
private:
    size_t rows, cols;
    std::map<std::pair<size_t, size_t>, T> data;

public:
    // To do same + Гаус
    SparseMatrix(size_t r, size_t c) : rows(r), cols(c) {}

    // Додати значення
    void set(size_t r, size_t c, const T& value) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Index out of range");
        }
        if (value != T()) {
            data[{r, c}] = value;
        }
        else {
            data.erase({ r, c });
        }
    }

    // Отримати значення
    T get(size_t r, size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Index out of range");
        }
        auto it = data.find({ r, c });
        if (it != data.end()) {
            return it->second;
        }
        return T();
    }

    // Виведення матриці
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << get(i, j) << " ";
            }
            std::cout << '\n';
        }
    }

    // Операція додавання
    SparseMatrix operator+(const SparseMatrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices dimensions must match for addition");
        }

        SparseMatrix result(rows, cols);
        for (const auto& [key, value] : data) {
            result.set(key.first, key.second, value + other.get(key.first, key.second));
        }
        for (const auto& [key, value] : other.data) {
            if (data.find(key) == data.end()) {
                result.set(key.first, key.second, value);
            }
        }
        return result;
    }

    // Операція віднімання
    SparseMatrix operator-(const SparseMatrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices dimensions must match for subtraction");
        }

        SparseMatrix result(rows, cols);
        for (const auto& [key, value] : data) {
            result.set(key.first, key.second, value - other.get(key.first, key.second));
        }
        for (const auto& [key, value] : other.data) {
            if (data.find(key) == data.end()) {
                result.set(key.first, key.second, -value);
            }
        }
        return result;
    }

    // Транспонування
    SparseMatrix transpose() const {
        SparseMatrix result(cols, rows);
        for (const auto& [key, value] : data) {
            result.set(key.second, key.first, value);
        }
        return result;
    }

    // Розв’язання системи рівнянь методом Гаусса
    std::vector<T> gaussSolve(const std::vector<T>& b) {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square to solve system of equations");
        }
        if (b.size() != rows) {
            throw std::invalid_argument("Vector size must match matrix dimensions");
        }

        SparseMatrix<T> mat = *this;
        std::vector<T> result = b;

        for (size_t i = 0; i < rows; ++i) {
            // Шукаємо головний елемент
            size_t maxRow = i;
            for (size_t k = i + 1; k < rows; ++k) {
                if (std::abs(static_cast<double>(mat.get(k, i))) > std::abs(static_cast<double>(mat.get(maxRow, i)))) {
                    maxRow = k;
                }
            }

            // Міняємо рядки місцями
            for (size_t j = 0; j < cols; ++j) {
                std::swap(mat.data[{i, j}], mat.data[{maxRow, j}]);
            }
            std::swap(result[i], result[maxRow]);

            // Нормалізація головного рядка
            T divisor = mat.get(i, i);
            for (size_t j = 0; j < cols; ++j) {
                mat.set(i, j, mat.get(i, j) / divisor);
            }
            result[i] = result[i] / divisor;

            // Виключення стовпця
            for (size_t k = i + 1; k < rows; ++k) {
                T factor = mat.get(k, i);
                for (size_t j = 0; j < cols; ++j) {
                    mat.set(k, j, mat.get(k, j) - factor * mat.get(i, j));
                }
                result[k] = result[k] - factor * result[i];
            }
        }

        // Зворотний хід
        std::vector<T> x(rows);
        for (int i = rows - 1; i >= 0; --i) {
            x[i] = result[i];
            for (size_t j = i + 1; j < cols; ++j) {
                x[i] = x[i] - mat.get(i, j) * x[j];
            }
        }

        return x;
    }
    // Обчислення детермінанта (метод Гаусса)
    T determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square to calculate determinant");
        }

        SparseMatrix<T> mat = *this;
        T det = 1;

        for (size_t i = 0; i < rows; ++i) {
            // Пошук головного елемента
            size_t maxRow = i;
            for (size_t k = i + 1; k < rows; ++k) {
                if (std::abs(static_cast<double>(mat.get(k, i))) > std::abs(static_cast<double>(mat.get(maxRow, i)))) {
                    maxRow = k;
                }
            }

            // Якщо головний елемент дорівнює нулю, детермінант нульовий
            if (mat.get(maxRow, i) == T()) {
                return T();
            }

            // Обмін рядків, якщо потрібно
            if (i != maxRow) {
                for (size_t j = 0; j < cols; ++j) {
                    std::swap(mat.data[{i, j}], mat.data[{maxRow, j}]);
                }
                det = -det;
            }

            // Множимо на головний елемент
            det *= mat.get(i, i);

            // Нормалізація головного рядка
            T divisor = mat.get(i, i);
            for (size_t j = 0; j < cols; ++j) {
                mat.set(i, j, mat.get(i, j) / divisor);
            }

            // Виключення стовпця
            for (size_t k = i + 1; k < rows; ++k) {
                T factor = mat.get(k, i);
                for (size_t j = 0; j < cols; ++j) {
                    mat.set(k, j, mat.get(k, j) - factor * mat.get(i, j));
                }
            }
        }

        return det;
    }

    // Computing norm 
    T norm() const {
        T maxNorm = T();
        for (size_t i = 0; i < rows; ++i) {
            T rowSum = T();
            for (size_t j = 0; j < cols; ++j) {
                rowSum += std::abs(static_cast<double>(get(i, j)));
            }
            maxNorm = std::max(maxNorm, rowSum);
        }
        return maxNorm;
    }
};

int main() {
    try {
        // choose matrix type
        std::cout << "Choose type: 1 - double, 2 - TRational: ";
        int choice;
        std::cin >> choice;

        if (choice == 1) {
            // matrix double
            Matrix<double> mat(3, 3);
            mat[0][0] = 1.0; mat[0][1] = 2.0; mat[0][2] = 3.0;
            mat[1][0] = 4.0; mat[1][1] = 5.0; mat[1][2] = 6.0;
            mat[2][0] = 7.0; mat[2][1] = 8.0; mat[2][2] = 9.0;

            std::cout << "Matrix double: \n";
            mat.print();

            auto transpose = mat.transpose();
            std::cout << "Transponed matrix: \n";
            transpose.print();

            auto determinant = mat.determinant();
            std::cout << "Det: " << determinant << "\n";

            auto norm = mat.norm();
            std::cout << "Norm: " << norm << "\n";

            // Gaussian
            std::vector<double> b = { 1.0, 2.0, 3.0 };
            auto solution = mat.gaussSolve(b);
            std::cout << "Gauss: ";
            for (const auto& x : solution) {
                std::cout << x << " ";
            }
            std::cout << "\n";

        }
        else if (choice == 2) {
            // matrix TRational
            Matrix<TRational> mat(2, 2);
            mat[0][0] = TRational(1, 2); mat[0][1] = TRational(3, 4);
            mat[1][0] = TRational(5, 6); mat[1][1] = TRational(7, 8);

            std::cout << "Matrix TRational: \n";
            mat.print();

            auto transpose = mat.transpose();
            std::cout << "Transponed matrix: \n";
            transpose.print();

            auto determinant = mat.determinant();
            std::cout << "Det: " << determinant << "\n";

            auto norm = mat.norm();
            std::cout << "Norm: " << norm << "\n";

            // Gaussian
            std::vector<TRational> b = { TRational(1, 1), TRational(2, 1) };
            auto solution = mat.gaussSolve(b);
            std::cout << "Gauss: ";
            for (const auto& x : solution) {
                std::cout << x << " ";
            }
            std::cout << "\n";
        }
        else {
            std::cerr << "Wrong choice!\n";
            return 1;
        }

        // SparseMatrix
        SparseMatrix<double> sparseMat(3, 3);
        sparseMat.set(0, 0, 1.0);
        sparseMat.set(1, 2, 2.5);
        sparseMat.set(2, 1, -3.0);

        std::cout << "Sparse matrix: \n";
        sparseMat.print();

        auto sparseTranspose = sparseMat.transpose();
        std::cout << "Transponed sparse matrix: \n";
        sparseTranspose.print();

        auto sparseDeterminant = sparseMat.determinant();
        std::cout << "Det sparse matrix: " << sparseDeterminant << "\n";

        auto sparseNorm = sparseMat.norm();
        std::cout << "Norm of sparse matrix: " << sparseNorm << "\n";

        std::vector<double> b = { 3.0, -2.0, 1.0 };
        auto sparseSolution = sparseMat.gaussSolve(b);
        std::cout << "Gauss for sparse matrix: ";
        for (const auto& x : sparseSolution) {
            std::cout << x << " ";
        }
        std::cout << "\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}