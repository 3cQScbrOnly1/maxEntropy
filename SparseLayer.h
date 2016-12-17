#include "tensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace std;

class SparseLayer
{
	private:
		// 一个二维矩阵，y' = Wx 中的W。这里的大小为 标签个数*特征个数 
		Tensor<cpu,2,double> _W; 
		// 存储_W 的损失 , 大小同_W 
		Tensor<cpu,2,double> _gradW; 
		// 大小同_W , 存储 _gradW 的平方求和
		Tensor<cpu,2,double> _eg2W;

		Tensor<cpu,2,double> _ftW;

		int _max_update;

		vector<int> _last_update;
	public:
		// 默认构造函数
		SparseLayer();  
	   	// 初始化为成员变量申请内存
		void inital(int o_size,int i_size);
	   	// y' = Wx 计算y'。
		void computeForwardScore(const vector<int> &x,Tensor<cpu,2,double> y);
	   	// 整个程序只有一层函数 不需要回传损失。 这里只有做了累积损失
		void computeBackwardLoss(const vector<int> &x,Tensor<cpu,2,double> ly);
		// 更新参数_W
		void updateW(const vector<int> &x); 
		//释放成员变量的内存
		~SparseLayer();

		void updateSparseWeight(int feature_id);

		void clearGradW(const vector<int> &x); 
};
