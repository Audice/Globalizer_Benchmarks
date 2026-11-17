#ifndef __RASTRIGINPROBLEM_H__
#define __RASTRIGINPROBLEM_H__

#include "IGlobalOptimizationProblem.h"

/** Задача Растригина с частично целочисленными параметрами

Число непрерывных и дискретных параметров задаётся параметрами задачи (#SetParameter,
в Globalizer - ключами командной строки, например -ContinuousVariables 3 -DiscreteVariables 2):
  - ContinuousVariables - число непрерывных параметров (не меньше 1);
  - DiscreteVariables   - число дискретных параметров (не меньше 0);
  - DiscreteValues      - число значений каждого дискретного параметра (от 2 до 26,
                          значения "A", "B", ...).
Размерность задачи равна ContinuousVariables + DiscreteVariables.

#SetDimension сохраняет число параметров, заданное явно: если задано только число дискретных,
меняется число непрерывных, иначе - число дискретных; если заданы оба, другую размерность
задать нельзя. Дискретные параметры всегда последние в векторе y.
*/
class RastriginIntProblem : public IGlobalOptimizationProblem
{
protected:
  /// Число непрерывных параметров
  int countContinuousVariables;
  /// Число дискретных параметров
  int countDiscreteVariables;
  /// Число непрерывных параметров задано явно (ContinuousVariables)
  bool mIsContinuousSet;
  /// Число дискретных параметров задано явно (DiscreteVariables)
  bool mIsDiscreteSet;

  /// Нижняя граница области поиска
  std::vector<double> A;
  /// Верхняя граница области поиска
  std::vector<double> B;
  /// Координаты точки оптимума (непрерывные и приведённые к double дискретные)
  std::vector<double> optPoint;
  /// Значения дискретных параметров ("A", "B", ...), приведённые к double
  std::vector<double> discreteValues;
  /// коэфициент для сдвига функции при добовление целочисленной части функции
  double multKoef;
  /// коэфициент сдвига значения целевой функции
  double optMultKoef;

  /// Размерность задачи
  int mDimension;
  /// Инициализирована ли задача
  bool mIsInitialized;
  /// Максимальная допустимая размерность задачи
  int mMaxDimension;
  /// минимальная  допустимая размерность задачи
  int mMinDimension;
  /// число критериев
  int mNumberOfCriterions;
  /// число ограничений
  int mNumberOfConstraints;
  /// Максимальная нижняя граница области поиска
  double mLeftBorder;
  /// Минимальная верхняя граница области поиска
  double mRightBorder;
  /// Число значений каждого дискретного параметра (DiscreteValues)
  int mDefNumberOfValues;
  /// Максимальное число значений дискретного параметра (буквы "A".."Z")
  int mMaxNumberOfValues;
  /// используется ли сдвиг
  bool IsMultInt;
  /// Число значений дискретных параметров по каждой размерности
  std::vector<int> mNumberOfValues;

  /// Задаёт число непрерывных и дискретных параметров; при недопустимых значениях ничего не меняет
  int SetVariables(int continuousCount, int discreteCount);

public:

  RastriginIntProblem();
  /** Метод возвращает значение целевой функции в точке глобального минимума
  \param[out] value оптимальное значение
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetOptimumValue(double& value) const;
  /** Метод возвращает координаты точки глобального минимума целевой функции
  \param[out] y точка, в которой достигается оптимальное значение
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetOptimumPoint(std::vector<double>& x, std::vector<std::string>& u) const;
  /** Метод, вычисляющий функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \param[in] u целочисленые координаты точки, в которой необходимо вычислить значение
  \param[in] fNumber Номер вычисляемой функции. 0 соответствует первому ограничению,
  #GetNumberOfFunctions() - 1 -- последнему критерию
  \return Значение функции с указанным номером
  */
  virtual double CalculateFunctionals(const std::vector<double>& x, std::vector<std::string>& u, int fNumber);

  /** Метод задаёт размерность задачи

  Сохраняет явно заданное число параметров (см. описание класса).
  \param[in] dimension размерность задачи
  \return Код ошибки
  */
  virtual int SetDimension(int dimension);
  /// Возвращает размерность задачи, можно вызывать после #Initialize
  virtual int GetDimension() const;

  /** Метод возвращает границы области поиска
  */
  virtual void GetBounds(std::vector<double>& lower, std::vector<double>& upper);

  /** Метод возвращает число общее функций в задаче (оно равно число ограничений + число критериев)
  \return Число функций
  */
  virtual int GetNumberOfFunctions() const;
  /** Метод возвращает число ограничений в задаче
  \return Число ограничений
  */
  virtual int GetNumberOfConstraints() const;
  /** Метод возвращает число критериев в задаче
  \return Число критериев
  */
  virtual int GetNumberOfCriterions() const;

  ~RastriginIntProblem();


  /// <summary>
  /// Вычисление множителя
  /// </summary>
  /// <param name="x"></param>
  /// <returns></returns>
  double MultFunc(const std::vector<double>& x);

  ///Инициализация задачи
  virtual int Initialize();

  /// Метод возвращает число дискретных параметров, дискретные параметры всегда последние в векторе y
  virtual int GetNumberOfDiscreteVariable() const;

  /** Метод задает число дискретных параметров при неизменной размерности,
  дискретные параметры всегда последние в векторе y
  \param[in] numberOfDiscreteVariable число дискретных параметров
  \return Код ошибки
  */
  virtual int SetNumberOfDiscreteVariable(int numberOfDiscreteVariable);


  /** Метод возвращает значения дискретных параметров
  \return Код ошибки
  */
  virtual int GetDiscreteVariableValues(std::vector< std::vector<std::string>>& values) const;

  using IGlobalOptimizationProblem::SetParameter;

  /** Метод задает параметры задачи: ContinuousVariables, DiscreteVariables, DiscreteValues
  \param[in] name имя параметра
  \param[in] value значение параметра (целое число)
  \return Код ошибки (#PROBLEM_OK; #PROBLEM_ERROR - недопустимое значение;
  #PROBLEM_UNDEFINED - неизвестный параметр)
  */
  virtual int SetParameter(std::string name, std::string value);

  /** Метод задает параметры задачи (значение int или string)
  */
  virtual int SetParameter(std::string name, IOptVariantType value);

  /** Метод возвращает параметры задачи и их текущие значения
  \param[out] names имена параметров
  \param[out] values значения параметров
  */
  virtual void GetParameters(std::vector<std::string>& names, std::vector<std::string>& values);

};

extern "C" LIB_EXPORT_API IGlobalOptimizationProblem* create();
extern "C" LIB_EXPORT_API void destroy(IGlobalOptimizationProblem* ptr);

#endif
// - end of file ----------------------------------------------------------------------------------
