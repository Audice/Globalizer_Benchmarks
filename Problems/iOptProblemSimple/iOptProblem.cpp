#ifndef WIN32
#include <dlfcn.h>
#endif

#include "iOptProblem.h"

#include <iostream>
#include <filesystem>
#include <variant>

// ------------------------------------------------------------------------------------------------
iOptProblem::iOptProblem()
{
  mIsInitialized = false;
  mDimension = 3;

  mPyFilePath = "";

  functionScriptName = "rastrigin";
  functionClassName = "Rastrigin";
  ProcRank = 0;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::SetConfigPath(const std::string& configPath)
{
  mPyFilePath = std::string(configPath);
  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::SetDimension(int dimension)
{
  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::GetDimension() const
{
  return mDimension;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::Initialize()
{
  if (!mIsInitialized)
  {
#ifndef WIN32
    mLibpython_handle = dlopen("/home/lebedev_i/miniconda3/lib/libpython3.9.so", RTLD_LAZY | RTLD_GLOBAL);
#endif

    if (mPyFilePath == "")
    {
      const char* filename = __FILE__;
      std::filesystem::path file_path(filename);
      mPyFilePath = file_path.parent_path().string();
    }
    GetParameters(problemParametersNames, problemParametersStringValues);

    mFunction = std::make_shared<TPythonModuleWrapper>(mPyFilePath, problemParametersValues, 
      problemParametersNames, functionScriptName, functionClassName);
    mDimension = mFunction->GetDimension();

    mIsInitialized = true;
    return IGlobalOptimizationProblem::PROBLEM_OK;
  }
  else
    return IGlobalOptimizationProblem::PROBLEM_ERROR;
}

int iOptProblem::GetNumberOfDiscreteVariable() const 
{
    return mFunction->GetDescreteParameters().size();
}

// ------------------------------------------------------------------------------------------------
void iOptProblem::GetBounds(std::vector<double>& lower, std::vector<double>& upper)
{
  if (mIsInitialized)
  {
    mFunction->GetBounds(lower, upper);
  }
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::GetOptimumValue(double& value) const
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::GetOptimumPoint(std::vector<double>& x, std::vector<std::string>& u) const
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::GetNumberOfFunctions() const
{
  if (mIsInitialized)
  {
    return mFunction->GetNumberOfFunctions();
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::GetNumberOfConstraints() const
{
  if (mIsInitialized)
  {
    return mFunction->GetNumberOfConstraints();
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------
int iOptProblem::GetNumberOfCriterions() const
{
  if (mIsInitialized)
  {
    return mFunction->GetNumberOfCriterions();
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------
double iOptProblem::CalculateFunctionals(const std::vector<double>& y, std::vector<std::string>& u, int fNumber)
{
  return mFunction->EvaluateFunction(y, u, fNumber);
}

// ------------------------------------------------------------------------------------------------
inline std::vector<double> iOptProblem::CalculateAllFunctionals(const std::vector<double>& y, std::vector<std::string>& u)
{
  return mFunction->EvaluateAllFunction(y, u);
}

inline int iOptProblem::GetStartTrial(std::vector<double>& y, std::vector<std::string>& u, std::vector<double>& values)
{
  if (mIsInitialized)
  {
    return mFunction->GetStartTrial(y, u, values);
  }
  return IGlobalOptimizationProblem::PROBLEM_ERROR;
}

inline int iOptProblem::GetDiscreteVariableValues(std::vector< std::vector<std::string>>& values) const
{
    values = mFunction->GetDescreteParameters();
    return IGlobalOptimizationProblem::PROBLEM_OK;
}

void iOptProblem::Finalize(const std::vector<double>& y, std::vector<std::string>& u)
{
    return mFunction->Finalize(y, u);
}


// ------------------------------------------------------------------------------------------------
inline int iOptProblem::SetParameter(std::string name, std::string value)
{
  if (name == "Dimension")
    mDimension = std::stoi(value);

  else if (name == "mPyFilePath")
    mPyFilePath = value;

  else if (name == "functionScriptName")
    functionScriptName = value;

  else   if (name == "functionClassName")
    functionClassName = value;

  else   if (name == "DataSet")
      datasetName = value;

  else   if (name == "Method")
      methodName = value;

  else   if (name == "ProcRank")
      ProcRank = std::stoi(value);;



  return IGlobalOptimizationProblem::PROBLEM_OK;
}

// ------------------------------------------------------------------------------------------------
inline int iOptProblem::SetParameter(std::string name, IOptVariantType value)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline int iOptProblem::SetParameter(std::string name, void* value)
{
  return IGlobalOptimizationProblem::PROBLEM_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
inline void iOptProblem::GetParameters(std::vector<std::string>& names, std::vector<std::string>& values)
{
  problemParametersNames.clear();
  problemParametersStringValues.clear();
  problemParametersValues.clear();

  problemParametersNames.push_back("Dimension");
  problemParametersStringValues.push_back(std::to_string(mDimension));
  problemParametersValues.push_back(mDimension);

  problemParametersNames.push_back("mPyFilePath");
  problemParametersStringValues.push_back(mPyFilePath);
  problemParametersValues.push_back(mPyFilePath);

  problemParametersNames.push_back("functionScriptName");
  problemParametersStringValues.push_back(functionScriptName);
  problemParametersValues.push_back(functionScriptName);

  problemParametersNames.push_back("functionClassName");
  problemParametersStringValues.push_back(functionClassName);
  problemParametersValues.push_back(functionClassName);

  problemParametersNames.push_back("DataSet");
  problemParametersStringValues.push_back(datasetName);
  problemParametersValues.push_back(datasetName);

  problemParametersNames.push_back("Method");
  problemParametersStringValues.push_back(methodName);
  problemParametersValues.push_back(methodName);
  
  problemParametersNames.push_back("ProcRank");
  problemParametersStringValues.push_back(std::to_string(ProcRank));
  problemParametersValues.push_back(ProcRank);

  names = problemParametersNames;
  values = problemParametersStringValues;
}

// ------------------------------------------------------------------------------------------------
iOptProblem::~iOptProblem()
{
#ifndef WIN32
  if (mLibpython_handle)
    dlclose(mLibpython_handle);
#endif
}

// ------------------------------------------------------------------------------------------------
LIB_EXPORT_API IGlobalOptimizationProblem* create()
{
  return new iOptProblem();
}

// ------------------------------------------------------------------------------------------------
LIB_EXPORT_API void destroy(IGlobalOptimizationProblem* ptr)
{
  delete ptr;
}
// - end of file ----------------------------------------------------------------------------------
