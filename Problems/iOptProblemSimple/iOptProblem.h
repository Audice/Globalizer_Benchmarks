#ifndef __IOPT_PROBLEM_H__
#define __IOPT_PROBLEM_H__

#include <memory>

#include "IGlobalOptimizationProblem.h"
#include "pymodule_wrapper.h"

/// Класс обертка для задач iOpt на python
class iOptProblem : public IGlobalOptimizationProblem
{
protected:

  /// Размерность задачи
  int mDimension;
  /// Путь до папки в которой лежат скрипты
  std::string mPyFilePath;
  /// Имя скрипта с задачей iOpt
  std::string functionScriptName;
  /// Имя класса с задачей iOpt
  std::string functionClassName;
  /// Инициализирована ли задача
  bool mIsInitialized;

  /// Имя скрипта с задачей iOpt
  std::string datasetName;
  /// Имя скрипта с задачей iOpt
  std::string methodName;

  /// Интерфейс для подключения задач на языке python
  std::shared_ptr<TPythonModuleWrapper> mFunction;
  /// Путь до python в linux
  void* mLibpython_handle;

  /// Имена параметров
  std::vector<std::string> problemParametersNames;
  /// Строковые значения параметров
  std::vector<std::string> problemParametersStringValues;
  /// Значения параметров
  std::vector<IOptVariantType> problemParametersValues;

  int ProcRank = 0;
public:

  iOptProblem();

  /** Задание пути до конфигурационного файла
  Данный метод должн вызываться перед #Initialize
  \param[in] configPath строка, содержащая путь к конфигурационному файлу задачи
  \return Код ошибки
  */
  virtual int SetConfigPath(const std::string& configPath);
  /** Метод задаёт размерность задачи

  Данный метод должен вызываться перед #Initialize. Размерность должна быть в
  списке поддерживаемых.
  \param[in] dimension размерность задачи
  \return Код ошибки
  */
  virtual int SetDimension(int dimension);
  /// Возвращает размерность задачи, можно вызывать после #Initialize
  virtual int GetDimension() const;
  ///Инициализация задачи
  virtual int Initialize();

  /** Метод возвращает границы области поиска
  */
  virtual void GetBounds(std::vector<double>& upper, std::vector<double>& lower);
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

  /** Метод, вычисляющий функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \param[in] u целочисленые координаты точки, в которой необходимо вычислить значение
  \param[in] fNumber Номер вычисляемой функции. 0 соответствует первому ограничению,
  #GetNumberOfFunctions() - 1 -- последнему критерию
  \return Значение функции с указанным номером
  */
  virtual double CalculateFunctionals(const std::vector<double>& y, std::vector<std::string>& u, int fNumber);
  /** Метод, вычисляющий все функции задачи

  \param[in] y непрерывные координаты точки, в которой необходимо вычислить значение
  \param[in] u целочисленые координаты точки, в которой необходимо вычислить значение
  \return Значение функций
  */
  virtual std::vector<double> CalculateAllFunctionals(const std::vector<double>& y, std::vector<std::string>& u);

  /** ћетод возвращает точку из допустимой области
  \param[out] y непрерывные координаты точки
  \param[out] u целочисленые координаты точки
  \param[out] values значение в этой точке
  \return  од ошибки (#PROBLEM_OK или #UNDEFINED)
  */
  virtual int GetStartTrial(std::vector<double>& y, std::vector<std::string>& u, std::vector<double>& values);

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

  /// Метод возвращает число дискретных параметров, дискретные параметры всегда последние в векторе y
  virtual int GetNumberOfDiscreteVariable() const;

  /** Метод возвращает число целочисленных переменных
  \return Число целочисленных переменных
  */
  int GetDiscreteVariableValues(std::vector< std::vector<std::string>>& values) const;

  virtual void Finalize(const std::vector<double>& y, std::vector<std::string>& u);

  ~iOptProblem();
};

extern "C" LIB_EXPORT_API IGlobalOptimizationProblem* create();
extern "C" LIB_EXPORT_API void destroy(IGlobalOptimizationProblem* ptr);

#endif
// - end of file ----------------------------------------------------------------------------------
