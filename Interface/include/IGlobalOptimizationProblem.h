/////////////////////////////////////////////////////////////////////////////
//                                                                         //
//             LOBACHEVSKY STATE UNIVERSITY OF NIZHNY NOVGOROD             //
//                                                                         //
//                       Copyright (c) 2025 by UNN.                        //
//                          All Rights Reserved.                           //
//                                                                         //
//  File:      IGlobalOptimizationProblem.h                                //
//                                                                         //
//  Purpose:   Header file for Globalizer problem interface                //
//                                                                         //
//                                                                         //
//  Author(s): Lebedev I.G.                                                //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

/**
\file IGlobalOptimizationProblem.h

\authors Лебедев И.Г.
\date 2025
\copyright ННГУ им. Н.И. Лобачевского

\brief Объявление абстрактного класса #IGlobalOptimizationProblem

\details Объявление абстрактного класса #IGlobalOptimizationProblem и сопутствующих типов данных
*/

#ifndef __I_GLOBAL_OPTIMIZATION_PROBLEM_H__
#define __I_GLOBAL_OPTIMIZATION_PROBLEM_H__

#include <vector>
#include <string>
#include <stdexcept>
#include <variant>
#include <iostream>

/// Варианты типов параметров задачи
using IOptVariantType = std::variant<int, double, std::string>;

/**
Базовый класс-интерфейс, от которого наследуются классы, описывающие задачи оптимизации.

В классе #IGlobalOptimizationProblem описаны прототипы методов, которые должны быть реализованы в подключамых модулях с задачами.
*/
class IGlobalOptimizationProblem
{
public:

  /// Код ошибки, возвращаемый, если операция завершена успешно
  static const int PROBLEM_OK = 0;

  /** Код ошибки, возвращаемый методами #GetOptimumValue и #GetOptimumPoint,
  если соответствующие параметры задачи не определены,
  */
  static const int PROBLEM_UNDEFINED = -1;

  /// Код ошибки, возвращаемый, если операция не выполнена
  static const int PROBLEM_ERROR = -2;

  /// Код ошибки, возвращаемый, если попытались получить значения для недискретного параметра
  static const int ERROR_DISCRETE_VALUE = -201;

  /** Метод, вычисляющий функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \param[in] u целочисленые координаты точки, в которой необходимо вычислить значение
  \param[in] fNumber Номер вычисляемой функции. 0 соответствует первому ограничению,
  #GetNumberOfFunctions() - 1 -- последнему критерию
  \return Значение функции с указанным номером
  */
  virtual double CalculateFunctionals(const std::vector<double>& y, std::vector<std::string>& u, int fNumber);

  /** Метод задаёт размерность задачи

  Данный метод должен вызываться перед #Initialize. Размерность должна быть в
  списке поддерживаемых.
  \param[in] dimension размерность задачи
  \return Код ошибки
  */
  virtual int SetDimension(int dimension) = 0;

  /// Возвращает размерность задачи, можно вызывать после #Initialize
  virtual int GetDimension() const = 0;

  ///Инициализация задачи
  virtual int Initialize() = 0;

  /** Метод возвращает границы области поиска
  */
  virtual void GetBounds(std::vector<double>& lower, std::vector<double>& upper) = 0;

  /** Метод возвращает значение целевой функции в точке глобального минимума
  \param[out] value оптимальное значение
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetOptimumValue(double& value) const = 0;

  /** Метод возвращает координаты точки глобального минимума целевой функции
  \param[out] y точка, в которой достигается оптимальное значение
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetOptimumPoint(std::vector<double>& y, std::vector<std::string>& u) const = 0;

  /** Метод возвращает число общее функций в задаче (оно равно число ограничений + число критериев)
  \return Число функций
  */
  virtual int GetNumberOfFunctions() const = 0;

  /** Метод возвращает число ограничений в задаче
  \return Число ограничений
  */
  virtual int GetNumberOfConstraints() const = 0;

  /** Метод возвращает число критериев в задаче
  \return Число критериев
  */
  virtual int GetNumberOfCriterions() const = 0;

  /** Задание пути до конфигурационного файла
  Данный метод должн вызываться перед #Initialize
  \param[in] configPath строка, содержащая путь к конфигурационному файлу задачи
  \return Код ошибки
  */
  virtual int SetConfigPath(const std::string& configPath);

  /** Метод возвращает значение функции с номером index в точке глобального минимума
  \param[out] value оптимальное значение
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetOptimumValue(double& value, int index) const;

  /** Метод возвращает координаты всех точек глобального минимума целевой функции
  и их количество
  \param[out] y непрерывные координаты точек, в которых достигается оптимальное значение
  \param[out] u целочисленные координаты точек, в которых достигается оптимальное значение
  \param[out] n количество точек, в которых достигается оптимальное значение
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetAllOptimumPoint(std::vector<std::vector<double>>& y, std::vector<std::vector<std::string>>& u, int& n) const;

  /** Метод возвращает точку из допустимой области
  \param[out] y непрерывные координаты точки
  \param[out] u целочисленые координаты точки
  \param[out] values значение в этой точке
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetStartTrial(std::vector<double>& y, std::vector<std::string>& u, std::vector<double>& values);

  /** Метод, вычисляющий функции задачи в нескольких точках одновременно
  \param[in] y массив, содержащий последовательно записанные непрерывные координаты точки, в которых необходимо
  вычислить функционалы задачи
  \param[in] u массив, содержащий последовательно записанные целочисленые координаты многомерные точки, в которых необходимо
  вычислить функционалы задачи
  \param[in] Номер вычисляемой функции
  \param[in] numPoints количество передаваемых точек
  \param[out] values массив, в который будут записаны вычисленные значения функционалов
  */
  virtual void CalculateFunctionals(std::vector<std::vector<double>>& y, std::vector<std::vector<std::string>>& u, 
    int fNumber, int& numPoints, std::vector<double>& values);

  /** Метод, вычисляющий все функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \param[in] u целочисленые координаты точки, в которой необходимо вычислить значение
  \return Значение функций
  */
  virtual std::vector<double> CalculateAllFunctionals(const std::vector<double>& y, std::vector<std::string>& u);

  /// Метод возвращает число дискретных параметров, дискретные параметры всегда последние в векторе y
  virtual int GetNumberOfDiscreteVariable() const;

  /** Метод задает число дискретных параметров, дискретные параметры всегда последние в векторе y, 
  только для задач с частично целочисленными параметрами
  \param[in] numberOfDiscreteVariable число дискретных параметров
  \return Код ошибки
  */
  virtual int SetNumberOfDiscreteVariable(int numberOfDiscreteVariable);

  /** Метод возвращает число целочисленных переменных
  \return Число целочисленных переменных
  */
  virtual int GetDiscreteVariableValues(std::vector< std::vector<std::string>>& values) const;

  /** Метод возвращает число непрерывных переменных
  \return Число непрерывных переменных
  */
  virtual int GetNumberOfContinuousVariable() const;

  /** Метод задает параметры задачи
  \param[in] name имя параметра
  \param[in] value значение параметра
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int SetParameter(std::string name, std::string value);

  /** Метод задает параметры задачи
  \param[in] name имя параметра
  \param[in] value значение параметра
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int SetParameter(std::string name, IOptVariantType value);

  /** Метод задает параметры задачи
  \param[in] name имя параметра
  \param[in] value значение параметра
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int SetParameter(std::string name, void* value);

  /** Метод возвращает параметры задачи
  \param[out] names имена параметраметров 
  \param[out] values значениея параметров
  \return Код ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual void GetParameters(std::vector<std::string>& names, std::vector<std::string>& values);

  virtual void Finalize(const std::vector<double>& y, std::vector<std::string>& u);

  ///Деструктор
  virtual ~IGlobalOptimizationProblem();
};

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::SetConfigPath(const std::string& configPath)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::GetOptimumValue(double& value, int index) const
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::GetAllOptimumPoint(std::vector<std::vector<double>>& y, std::vector<std::vector<std::string>>& u, int& n) const
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

inline int IGlobalOptimizationProblem::GetStartTrial(std::vector<double>& y, std::vector<std::string>& u, std::vector<double>& values)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline double IGlobalOptimizationProblem::CalculateFunctionals(const std::vector<double>& y, std::vector<std::string>& u, int fNumber)
{
  throw std::runtime_error(std::string("Required overload of the following method is not implemented: ")
    + std::string(__FUNCTION__));
}

// ------------------------------------------------------------------------------------------------
inline void IGlobalOptimizationProblem::CalculateFunctionals(std::vector<std::vector<double>>& y, std::vector<std::vector<std::string>>& u,
  int fNumber, int& numPoints, std::vector<double>& values)
{
  throw std::runtime_error(std::string("Required overload of the following method is not implemented: ")
    + std::string(__FUNCTION__));
}

// ------------------------------------------------------------------------------------------------
inline std::vector<double> IGlobalOptimizationProblem::CalculateAllFunctionals(const std::vector<double>& y, std::vector<std::string>& u)
{
  throw std::runtime_error(std::string("Required overload of the following method is not implemented: ")
    + std::string(__FUNCTION__));
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::GetNumberOfDiscreteVariable() const
{
  return 0;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::GetNumberOfContinuousVariable() const
{
  return GetDimension() - GetNumberOfDiscreteVariable();
}


// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::SetNumberOfDiscreteVariable(int numberOfDiscreteVariable)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::GetDiscreteVariableValues(std::vector< std::vector<std::string>>& values) const
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::SetParameter(std::string name, std::string value)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::SetParameter(std::string name, IOptVariantType value)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int IGlobalOptimizationProblem::SetParameter(std::string name, void* value)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline void IGlobalOptimizationProblem::GetParameters(std::vector<std::string>& names, std::vector<std::string>& values)
{
   
}

inline void IGlobalOptimizationProblem::Finalize(const std::vector<double>& y, std::vector<std::string>& u)
{
    std::cout << "Finalize not implemented!" << std::endl;
}

// ------------------------------------------------------------------------------------------------
inline IGlobalOptimizationProblem::~IGlobalOptimizationProblem() {}

///Тип функции-фабрики, которая экспортируется подключаемой библиотекой с задачей
typedef IGlobalOptimizationProblem* createProblem();
///Тип функции-деструктора, которая экспортируется подключаемой библиотекой с задачей
typedef void destroyProblem(IGlobalOptimizationProblem*);

///Префикс для фуккций, экспортируемых подключаемой библиотекой с задачей
#ifdef WIN32
#define LIB_EXPORT_API __declspec(dllexport)
#else
#define LIB_EXPORT_API
#endif

#endif
// - end of file ----------------------------------------------------------------------------------
