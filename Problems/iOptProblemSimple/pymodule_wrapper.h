#pragma once

#include <string>
#include <vector>
#include <variant>

#include "IGlobalOptimizationProblem.h"

#ifndef WIN32
  #ifdef _DEBUG
    #undef _DEBUG
    #include "Python.h"
    #define _DEBUG
  #else
    #include "Python.h"
  #endif
#else
  #ifdef _DEBUG
    #undef _DEBUG
    #include "python.h"
    #define _DEBUG
  #else
    #include "python.h"
  #endif
#endif


/// Интерфейс для подключения задач iOpt на языке python
class TPythonModuleWrapper
{
protected:
  /// Размерность задачи
  int mDimension;

  /// Список дискретных параметров
  std::vector<std::vector<std::string>> discreteParams;

  /// Верхняя граница области поиска
  std::vector<double> mUpperBound;
  /// Нижняя граница области поиска
  std::vector<double> mLowerBound;

  /// Экземпляр класса GlobalizerProblem
  PyObject* pInstance;
  /// класс GlobalizerProblem
  PyObject* pClass;
  /// скрипт Globalizer_problem
  PyObject* pModule;
  /// Задача из iOpt
  PyObject* funcClass;

  PyThreadState* main_ts;

  
  /// <summary>
  /// Создает параметры для создания объекта задачи iOpt
  /// </summary>
  /// <param name="param"></param>
  /// <returns></returns>
  PyObject* VectorToTuple(std::vector<IOptVariantType> param,
    std::vector<std::string> paramName, std::vector<std::string> mProblemParametersNames);

public:
  /// <summary>
  /// 
  /// </summary>
  /// <param name="module_path"> путь до папки в которой лежат скрипты </param>
  /// <param name="param"> вектор параметров для создания задачи</param>
  /// <param name="functionScriptName">имя скрипта с задачей iOpt</param>
  /// <param name="functionClassName">имя класса с задачей iOpt</param>
  TPythonModuleWrapper(const std::string& module_path, std::vector<IOptVariantType> param = {}, 
    std::vector<std::string> paramName = {},
    std::string functionScriptName = "rastrigin", std::string functionClassName= "Rastrigin");

  /// Возвращает размерность задачи, можно вызывать после #Initialize
  int GetDimension() const;
  /** Метод возвращает границы области поиска
  */
  void GetBounds(std::vector<double>& lower, std::vector<double>& upper) const;
  /** Метод, вычисляющий функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \param[in] fNumber Номер вычисляемой функции. 0 соответствует первому ограничению,
  #GetNumberOfFunctions() - 1 -- последнему критерию
  \return Значение функции с указанным номером
  */
  double EvaluateFunction(const std::vector<double>& y, const std::vector<std::string>& categorys, int fNumber) const;


  virtual void Finalize(const std::vector<double>& y, std::vector<std::string>& categorys);

  /** Метод, вычисляющий все функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \return Значение функций
  */
  std::vector<double> EvaluateAllFunction(const std::vector<double>& y, const std::vector<std::string>& categorys) const;

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

  /** Метод возвращает список дискретных параметров
    \return вектор дискретных параметров
    */
  virtual std::vector<std::vector<std::string>> GetDescreteParameters();

  /** ћетод возвращает точку из допустимой области
  \param[out] y непрерывные координаты точки
  \param[out] u целочисленые координаты точки
  \param[out] values значение в этой точке
  \return  од ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetStartTrial(std::vector<double>& y, std::vector<std::string>& u, std::vector<double>& values);

  ~TPythonModuleWrapper();
};
