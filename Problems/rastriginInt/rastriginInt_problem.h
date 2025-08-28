#ifndef __RASTRIGINPROBLEM_H__
#define __RASTRIGINPROBLEM_H__

#include "IGlobalOptimizationProblem.h"

class RastriginIntProblem : public IGlobalOptimizationProblem
{


protected:
  int countContinuousVariables;

  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> optPoint;
  double multKoef;
  double optMultKoef;

  int mDimension;
  bool mIsInitialized;
  int mMaxDimension;

  int mMinDimension;
  int mNumberOfCriterions;
  int mNumberOfConstraints;
  double mLeftBorder;
  double mRightBorder;

  int mDefNumberOfValues;

  bool IsMultInt;

  std::vector<int> mNumberOfValues;

  /// ������� ����� �������� �������� ��� ����������� ���������
  virtual void ClearCurrentDiscreteValueIndex(int** mCurrentDiscreteValueIndex)
  {
    if (GetNumberOfDiscreteVariable() > 0)
    {
      if (*mCurrentDiscreteValueIndex != 0)
        delete[] * mCurrentDiscreteValueIndex;

      *mCurrentDiscreteValueIndex = new int[GetNumberOfDiscreteVariable()];
      for (int i = 0; i < GetNumberOfDiscreteVariable(); i++)
        (*mCurrentDiscreteValueIndex)[i] = 0;
    }
  }

public:

  RastriginIntProblem();

  virtual int GetOptimumValue(double& value) const;
  virtual int GetOptimumPoint(std::vector<double>& x) const;

  virtual double CalculateFunctionals(const std::vector<double>& x, int fNumber);


  virtual int SetDimension(int dimension);
  virtual int GetDimension() const;


  virtual void GetBounds(std::vector<double>& lower, std::vector<double>& upper);

  virtual int GetNumberOfFunctions() const;
  virtual int GetNumberOfConstraints() const;
  virtual int GetNumberOfCriterions() const;

  ~RastriginIntProblem();



  double MultFunc(const std::vector<double>& x);

  ///������������� ������
  virtual int Initialize();

  /// ����� ���������� ����� ���������� ����������, ���������� ��������� ������ ��������� � ������� y
  virtual int GetNumberOfDiscreteVariable() const;

  /** ����� ������ ����� ���������� ����������, ���������� ��������� ������ ��������� � ������� y,
  ������ ��� ����� � �������� �������������� �����������
  \param[in] numberOfDiscreteVariable ����� ���������� ����������
  \return ��� ������
  */
  virtual int SetNumberOfDiscreteVariable(int numberOfDiscreteVariable);


  /** ����� ���������� ����� ������������� ����������
  \return ����� ������������� ����������
  */
  virtual int GetDiscreteVariableValues(std::vector< std::vector<double>> values) const;

};

extern "C" LIB_EXPORT_API IGlobalOptimizationProblem* create();
extern "C" LIB_EXPORT_API void destroy(IGlobalOptimizationProblem* ptr);

#endif
// - end of file ----------------------------------------------------------------------------------
