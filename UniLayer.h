#include "tensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace std;

class UniLayer 
{
	private:
		// 一个二维矩阵，y' = Wx 中的W。这里的大小为 标签个数*特征个数 
		Tensor<cpu,2,double> _W; 
		// 存储_W 的损失 , 大小同_W 
		Tensor<cpu,2,double> _gradW; 
		// 大小同_W , 存储 _gradW 的平方求和
		Tensor<cpu,2,double> _eg2W;

	public:
		// 默认构造函数
		UniLayer();  
	   	// 初始化为成员变量申请内存
		void initial(int o_size,int i_size);
	   	// y' = Wx 计算y'。
		void computeForwardScore(Tensor<cpu,2,double> x,Tensor<cpu,2,double> y);
	   	// 整个程序只有一层函数 不需要回传损失。 这里只有做了累积损失
		void computeBackwardLoss(Tensor<cpu,2,double> x,Tensor<cpu,2,double> ly);
		// 更新参数_W
		void updateW(); 
		//释放成员变量的内存
		~UniLayer();
};
