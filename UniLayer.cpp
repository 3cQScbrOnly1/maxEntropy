#include "UniLayer.h"
#include "Utiltensor.h"

#define ADAALPA 0.01
#define ADAEPS 1e-6
#define REG 1e-8

// 神马也不做
UniLayer::UniLayer()
{}

void UniLayer::initial(int o_size,int i_size)
{
	double bound = sqrt(6 / (o_size + i_size + 1));	
	// 为_W申请内存 
	_W = NewTensor<cpu>(Shape2(o_size,i_size),0.0); 
	
	random(_W,-1.0 * bound,1.0 * bound,0);
	// 为_gradW申请内存 
	_gradW = NewTensor<cpu>(Shape2(o_size,i_size),0.0); 
	// 为_eg2W申请内存
	_eg2W = NewTensor<cpu>(Shape2(o_size,i_size),0.0);
	// 为_ftW申请内存
}

void UniLayer::computeForwardScore(Tensor<cpu,2,double> x,Tensor<cpu,2,double> y)
{
	// 矩阵相乘 y=Wx
	y = dot(x,_W.T());
}

void UniLayer::computeBackwardLoss(Tensor<cpu,2,double> x,Tensor<cpu,2,double> ly)
{
	// 累积损失
	_gradW += dot(ly.T(),x);
}

void UniLayer::updateW()
{
	// 公式
	_gradW += _W * REG;		
	_eg2W +=  _gradW * _gradW;
	_W -= (_gradW * ADAALPA / F<nl_sqrt>(_eg2W + ADAEPS));
	// 重置_gradW
	_gradW = 0;
}

UniLayer::~UniLayer()
{
	//释放_W内存
	FreeSpace(&_W);
	//释放_gradW内存
	FreeSpace(&_gradW);
	//释放_eg2W内存
	FreeSpace(&_eg2W);
}

