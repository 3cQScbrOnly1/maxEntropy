#include "Example.h"
#include "tensor.h"
#include "Utiltensor.h"
#include "UniLayer.h"

#include <iostream>
#include <vector>

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace std;

class WClassifier
{
	private:
		// 标签个数 (最后结果分多少类)
		int _labels_size;
		// 词向量
		Tensor<cpu,2,double> _words;
		// 只有一层函数
		UniLayer _layer;
		// 词向量维度
		int _dim_size;
		//  y'=Wx 算出的预测结果output, (这里output 就是y')对其计算损失
		void softmaxLoss(Tensor<cpu,2,double> output,const vector<int> &answer,Tensor<cpu,2,double> loutput);
		// 用 y'=Wx 算出的预测结果output，(这里output 就是y')来看句子分成了哪一类。这个函数返回一个下标，去标签表里可以查具体结果
		int predictByOutput(Tensor<cpu,2,double> output);
		// 向量平均
		void avgWordEmb(Tensor<cpu,2,double> words_prime,Tensor<cpu,2,double> avg);
		// max向量
		void maxWordEmb(Tensor<cpu,2,double> words_prime,Tensor<cpu,2,double> max);
		// min向量
		void minWordEmb(Tensor<cpu,2,double> words_prime,Tensor<cpu,2,double> min);
		// 三个向量合成一个
		void concat(Tensor<cpu,2,double> min,Tensor<cpu,2,double> avg,Tensor<cpu,2,double> max,Tensor<cpu,2,double> pool_merge);
	public:
		// 默认构造函数，反正不调用
		WClassifier(); 
		// 构造函数 用于初始化矩阵运算库和_W
		WClassifier(int labels_size,NRMat<double> &word_emb);
	   	// 训练 _W 
		void myTrain(vector<Example> train_examples);
		// 对测试数据进行预测结果 
		void predict(vector<Example> test_examples); 
		// 析构函数 关闭矩阵运算和归还内存
		~WClassifier(); 

};
