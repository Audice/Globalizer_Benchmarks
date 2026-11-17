#include "rastriginInt_problem.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <stdexcept>

// ------------------------------------------------------------------------------------------------
RastriginIntProblem::RastriginIntProblem()
{
  this->mIsInitialized = false;
  mMaxDimension = 100;
  mMinDimension = 2;
  mNumberOfCriterions = 1;
  mNumberOfConstraints = 0;
  mLeftBorder = -2.2;
  mRightBorder = 1.8;
  countContinuousVariables = 1;
  countDiscreteVariables = 1;
  mIsContinuousSet = false;
  mIsDiscreteSet = false;
  mDimension = countContinuousVariables + countDiscreteVariables;
  mDefNumberOfValues = 2;
  mMaxNumberOfValues = 26;

  multKoef = 0;
  optMultKoef = 0;
  IsMultInt = true;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::SetVariables(int continuousCount, int discreteCount)
{
  // хотя бы один непрерывный параметр
  if (continuousCount < 1 || discreteCount < 0 || continuousCount + discreteCount > mMaxDimension)
    return IGlobalOptimizationProblem::PROBLEM_ERROR;

  countContinuousVariables = continuousCount;
  countDiscreteVariables = discreteCount;
  mDimension = continuousCount + discreteCount;
  if (mIsInitialized)
    return Initialize();
  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetOptimumValue(double& value) const
{
  if (!this->mIsInitialized)
    return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;

  std::vector<double> point;
  std::vector<std::string> u;

  GetOptimumPoint(point, u);

  value = const_cast<RastriginIntProblem*>(this)->CalculateFunctionals(point, u, 0);

  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetOptimumPoint(std::vector<double>& point, std::vector<std::string>& u) const
{
  if (!this->mIsInitialized)
    return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;

  point.assign(countContinuousVariables, 0.0);

  auto ndv = GetNumberOfDiscreteVariable();

  u.resize(ndv);

  std::vector< std::vector<std::string>> values;
  GetDiscreteVariableValues(values);

  // оптимум по дискретным параметрам - последнее значение (наибольшее, равное mRightBorder)
  for (int i = 0; i < ndv; i++)
    u[i] = values[i][values[i].size() - 1];
  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
double RastriginIntProblem::CalculateFunctionals(const std::vector<double>& x, std::vector<std::string>& u, int fNumber)
{
  if ((int)x.size() < countContinuousVariables || (int)u.size() < countDiscreteVariables)
    throw std::invalid_argument("rastriginInt: expected " + std::to_string(countContinuousVariables) +
      " continuous and " + std::to_string(countDiscreteVariables) + " discrete variables");

  double sum = 0.;
  int j = 0;
  int i = 0;
  std::vector<double> y(mDimension);
  for (; i < countContinuousVariables; i++)
    y[i] = x[i];
  for (; i < mDimension; i++)
  {
    const std::string& ui = u[i - countContinuousVariables];
    int k = ui.empty() ? -1 : ui[0] - 'A';
    if (ui.size() != 1 || k < 0 || k >= (int)discreteValues.size())
      throw std::invalid_argument("rastriginInt: invalid discrete value '" + ui + "'");
    y[i] = discreteValues[k];
  }
  for (; j < countContinuousVariables; j++)
    sum += y[j] * y[j] - 10. * cos(2.0 * M_PI * y[j]) + 10.0;
  for (; j < mDimension; j++)
  {
    sum = sum - y[j];
  }

  sum = sum * (MultFunc(y) + multKoef);

  return sum;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::SetDimension(int dimension)
{
  int res = IGlobalOptimizationProblem::PROBLEM_OK;
  if (dimension != mDimension)
  {
    if (mIsContinuousSet && mIsDiscreteSet)
      return IGlobalOptimizationProblem::PROBLEM_ERROR;   // размерность задана параметрами

    if (mIsDiscreteSet)                                   // сохраняем число дискретных параметров
      res = SetVariables(dimension - countDiscreteVariables, countDiscreteVariables);
    else                                                  // иначе - число непрерывных
      res = SetVariables(countContinuousVariables, dimension - countContinuousVariables);
  }
  if (res == IGlobalOptimizationProblem::PROBLEM_OK && !mIsInitialized)
    res = Initialize();
  return res;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetDimension() const
{
  return mDimension;
}

// ------------------------------------------------------------------------------------------------
void RastriginIntProblem::GetBounds(std::vector<double>& lower, std::vector<double>& upper)
{
  if (mIsInitialized)
  {
    lower.resize(mDimension);
    upper.resize(mDimension);

    for (int i = 0; i < mDimension; i++)
    {
      lower[i] = mLeftBorder;
      upper[i] = mRightBorder;
    }
  }
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetNumberOfFunctions() const
{
  return mNumberOfConstraints + mNumberOfCriterions;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetNumberOfConstraints() const
{
  return mNumberOfConstraints;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetNumberOfCriterions() const
{
  return mNumberOfCriterions;
}

// ------------------------------------------------------------------------------------------------
RastriginIntProblem::~RastriginIntProblem()
{

}


// ------------------------------------------------------------------------------------------------
double RastriginIntProblem::MultFunc(const std::vector<double>& x)
{
  double res = 0;

  double a;
  double d;

  for (int j = 0; j < this->mDimension; j++)
  {
    d = (B[j] - A[j]) / 2;
    a = (x[j] - optPoint[j]) / d;
    a = a * a;
    res = res + a;
  }

  res = -res;

  if (IsMultInt == true)
    return res;
  else
    return 1.0;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::Initialize()
{
  this->mIsInitialized = true;
  A.resize(this->mDimension);
  B.resize(this->mDimension);
  this->GetBounds(A, B);

  multKoef = 0;
  optMultKoef = 0;

  mNumberOfValues.assign(GetNumberOfDiscreteVariable(), mDefNumberOfValues);

  discreteValues.resize(mDefNumberOfValues);
  double d = (mRightBorder - mLeftBorder) / (mDefNumberOfValues - 1);
  for (int i = 0; i < mDefNumberOfValues; i++)
  {
    discreteValues[i] = mLeftBorder + d * i;
  }

  // точка оптимума: непрерывные координаты, затем значения дискретных
  std::vector<std::string> u;
  this->GetOptimumPoint(optPoint, u);
  for (const auto& ui : u)
    optPoint.push_back(discreteValues[ui[0] - 'A']);

  // max |MultFunc| по вершинам области: слагаемые независимы, поэтому максимум суммы равен
  // сумме максимумов по координатам (без перебора 2^mDimension вершин)
  for (int j = 0; j < this->mDimension; j++)
  {
    double h = (B[j] - A[j]) / 2;
    double a = (A[j] - optPoint[j]) / h;
    double b = (B[j] - optPoint[j]) / h;
    multKoef += std::max(a * a, b * b);
  }
  if (IsMultInt != true)
    multKoef = 1.0;
  multKoef += 4;
  optMultKoef = (MultFunc(optPoint) + multKoef);

  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetNumberOfDiscreteVariable() const
{
  return countDiscreteVariables;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::SetNumberOfDiscreteVariable(int numberOfDiscreteVariable)
{
  int res = SetVariables(mDimension - numberOfDiscreteVariable, numberOfDiscreteVariable);
  if (res == IGlobalOptimizationProblem::PROBLEM_OK && !mIsInitialized)
    res = Initialize();
  return res;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::GetDiscreteVariableValues(std::vector< std::vector<std::string>>& values) const
{
  values.resize(GetNumberOfDiscreteVariable());

  for (int i = 0; i < GetNumberOfDiscreteVariable(); i++)
  {
    int count = i < (int)mNumberOfValues.size() ? mNumberOfValues[i] : mDefNumberOfValues;
    values[i].resize(count);
    for (int j = 0; j < count; j++)
    {
      values[i][j] = std::string(1, 'A' + j);
    }
  }
  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::SetParameter(std::string name, std::string value)
{
  if (name != "ContinuousVariables" && name != "DiscreteVariables" && name != "DiscreteValues")
    return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;

  int v = 0;
  try
  {
    size_t pos = 0;
    v = std::stoi(value, &pos);
    if (value.find_first_not_of(" \t", pos) != std::string::npos)
      return IGlobalOptimizationProblem::PROBLEM_ERROR;
  }
  catch (const std::exception&)
  {
    return IGlobalOptimizationProblem::PROBLEM_ERROR;
  }

  int res = IGlobalOptimizationProblem::PROBLEM_OK;
  if (name == "ContinuousVariables")
  {
    res = SetVariables(v, countDiscreteVariables);
    if (res == IGlobalOptimizationProblem::PROBLEM_OK)
      mIsContinuousSet = true;
  }
  else if (name == "DiscreteVariables")
  {
    res = SetVariables(countContinuousVariables, v);
    if (res == IGlobalOptimizationProblem::PROBLEM_OK)
      mIsDiscreteSet = true;
  }
  else // DiscreteValues
  {
    if (v < 2 || v > mMaxNumberOfValues)
      return IGlobalOptimizationProblem::PROBLEM_ERROR;
    mDefNumberOfValues = v;
    if (mIsInitialized)
      res = Initialize();
  }
  return res;
}

// ------------------------------------------------------------------------------------------------
int RastriginIntProblem::SetParameter(std::string name, IOptVariantType value)
{
  if (std::holds_alternative<int>(value))
    return SetParameter(name, std::to_string(std::get<int>(value)));
  if (std::holds_alternative<std::string>(value))
    return SetParameter(name, std::get<std::string>(value));
  return IGlobalOptimizationProblem::PROBLEM_ERROR;   // число параметров - целое
}

// ------------------------------------------------------------------------------------------------
void RastriginIntProblem::GetParameters(std::vector<std::string>& names, std::vector<std::string>& values)
{
  // "Dimension" здесь нет: Globalizer передал бы задаче значение ключа -Dimension по умолчанию
  names = { "ContinuousVariables", "DiscreteVariables", "DiscreteValues" };
  values = { std::to_string(countContinuousVariables), std::to_string(countDiscreteVariables),
    std::to_string(mDefNumberOfValues) };
}

// ------------------------------------------------------------------------------------------------
LIB_EXPORT_API IGlobalOptimizationProblem* create()
{
  return new RastriginIntProblem();
}

// ------------------------------------------------------------------------------------------------
LIB_EXPORT_API void destroy(IGlobalOptimizationProblem* ptr)
{
  delete ptr;
}
// - end of file ----------------------------------------------------------------------------------
