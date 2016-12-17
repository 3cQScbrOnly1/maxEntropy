#include "SparseLayer.h" 
#include "Utiltensor.h" 
#define ADAALPA 0.01
#define ADAEPS 1e-6
#define REG 1e-8

// 神马也不做
SparseLayer::SparseLayer()
{}

void SparseLayer::inital(int o_size,int i_size)
{
	double bound = sqrt(6 / (o_size + i_size + 1));	
	// 为_W申请内存 
	_W = NewTensor<cpu>(Shape2(i_size,o_size),0.0); 
	
	random(_W,-1.0 * bound,1.0 * bound,0);
	// 为_gradW申请内存 
	_gradW = NewTensor<cpu>(Shape2(i_size,o_size),0.0); 
	// 为_eg2W申请内存
	_eg2W = NewTensor<cpu>(Shape2(i_size,o_size),0.0);
	// 为_ftW申请内存
	_ftW = NewTensor<cpu>(Shape2(i_size,o_size),0.0);
	// 初始化 _max_update
	int _max_update = 0;
	// 初始化_last_update
	for(int i = 0; i < i_size; i++)
		_last_update.push_back(0);
}

void SparseLayer::computeForwardScore(const vector<int> &x,Tensor<cpu,2,double> y)
{
	// x 中存了特征在特征表中的下标，做稀疏矩阵乘法。
	for(int i = 0; i < x.size(); i++)
	{
		updateSparseWeight(x[i]);
		y[0] += _W[x[i]];
	}

}

void SparseLayer::updateSparseWeight(int feature_id)
{
	if(_last_update[feature_id] < _max_update)	
	{
		int times = _max_update - _last_update[feature_id];
		_W[feature_id] *= F<nl_exp>(times * F<nl_log>(_ftW[feature_id]));
		_last_update[feature_id] = _max_update;
	}
}


void SparseLayer::computeBackwardLoss(const vector<int> &x,Tensor<cpu,2,double> ly)
{
	// 损失累积到_gradW
	for(int i = 0; i < x.size(); i++)
		_gradW[x[i]] += ly[0];
}

void SparseLayer::updateW(const vector<int> &x)
{
	_max_update++;
	// 按照公式更新_W
	for(int i = 0; i < x.size(); i++)	
	{
		// 按照特征下标去更新_W
		int feature_id = x[i];
		_eg2W[feature_id] += _gradW[feature_id] * _gradW[feature_id];
		Tensor<cpu,1,double> sqrt_eg2w = NewTensor<cpu>(Shape1(_W.size(1)),0.0);
		sqrt_eg2w = F<nl_sqrt>(_eg2W[feature_id] + ADAEPS);
		_W[feature_id] = (_W[feature_id] * sqrt_eg2w - _gradW[feature_id] * ADAALPA) / (ADAALPA * REG + sqrt_eg2w);

		_ftW[feature_id] = sqrt_eg2w / (ADAALPA * REG + sqrt_eg2w);
	}
	clearGradW(x);
}

void SparseLayer::clearGradW(const vector<int> &x)
{
	// 重置_gradW
	for(int i = 0; i < x.size(); i++)
		_gradW[x[i]] = 0;
}

SparseLayer::~SparseLayer()
{
	// 释放_W 内存 
	FreeSpace(&_W);  
	// 释放_gradW 内存 
	FreeSpace(&_gradW);
	// 释放_eg2W
	FreeSpace(&_eg2W);
	// 释放_ftW
	FreeSpace(&_ftW);
}
