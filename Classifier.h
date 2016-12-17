#include "Example.h"
#include "tensor.h"
#include "SparseLayer.h"

#include <iostream>
#include <vector>

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace std;



class Classifier
{
	private:
		//标签个数 (最后结果分多少类)
		int _labels_size; 
		// 特征个数
		int _features_size; 
		// 只有一层函数
		SparseLayer _layer;
		//  y'=Wx 算出的预测结果output, (这里output 就是y')对其计算损失
		void softmaxLoss(Tensor<cpu,2,double> output,const vector<int> &answer,Tensor<cpu,2,double> loutput);
		// 用 y'=Wx 算出的预测结果output，(这里output 就是y')来看句子分成了哪一类。这个函数返回一个下标，去标签表里可以查具体结果
		int predictByOutput(Tensor<cpu,2,double> output);
	public:
		// 默认构造函数，反正不调用
		Classifier(); 
		// 构造函数 用于初始化矩阵运算库和_W
		Classifier(int labels_size,int features_size);
	   	// 训练 _W 
		void myTrain(vector<Example> train_examples);
		// 对测试数据进行预测结果 
		void predict(vector<Example> test_examples); 
		// 析构函数 关闭矩阵运算和归还内存
		~Classifier(); 
};
