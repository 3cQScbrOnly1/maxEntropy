#include "Classifier.h"
#include "Counter.h"

#include <cmath>

// 神马也不做
Classifier::Classifier()
{}

Classifier::Classifier(int labels_size,int features_size)
{
	// 初始化矩阵运算 用cpu计算
	InitTensorEngine<cpu>(); 
	// 为了使类内可以访问标签个数
	this -> _labels_size = labels_size;
	// 为了使类内可以访问特征个数
	this -> _features_size = features_size; 
	//初始化_layer
	_layer.inital(_labels_size,_features_size);
}

void Classifier::myTrain(vector<Example> train_examples)
{
	vector<int> random_pos;
	for(int i = 0; i< train_examples.size(); i++)
		random_pos.push_back(i);
	random_shuffle(random_pos.begin(),random_pos.end());
	// 遍历每一个训练的句子
	for(int i = 0; i < train_examples.size(); i++)			
	{
		int pos = random_pos[i];
		// output存储 用_layer预测后的结果 
		// output_loss 存储预测结果的损失
		Tensor<cpu,2,double> output,output_loss;

		// 为output output_loss 申请内存
		output = NewTensor<cpu>(Shape2(1,_labels_size),0.0);	
		output_loss = NewTensor<cpu>(Shape2(1,_labels_size),0.0);	

		_layer.computeForwardScore(train_examples[pos]._d_features,output);
		softmaxLoss(output,train_examples[pos]._d_label,output_loss);
		_layer.computeBackwardLoss(train_examples[pos]._d_features,output_loss);
		_layer.updateW(train_examples[pos]._d_features);

		// 释放output output_loss 内存	
		FreeSpace(&output);
	   	FreeSpace(&output_loss); 
	}
}

void Classifier::softmaxLoss(Tensor<cpu,2,double> output,const vector<int> &answer,Tensor<cpu,2,double> loutput)
{
	// 这里和下面的PredictByOutput 类似。 同样也是找到output 中哪一维的数字最大
	int my_answer_pos = 0;		
	for(int i = 1; i < _labels_size; i++)
		if(output[0][i] > output[0][my_answer_pos])
			my_answer_pos = i;
	// scores 是将output概率化所需要的变量
	Tensor<cpu,2,double> scores = NewTensor<cpu>(Shape2(1,_labels_size),0.0);
	//sum 存储scores 求和
	double sum = 0;
	// output 中的最大数字存入max_score
	double max_score = output[0][my_answer_pos];
	// 遍历 output
	for(int i = 0; i < _labels_size; i++)	
	{
		// 概率化 output
		scores[0][i] = exp(output[0][i] - max_score);
		// output 所有概率化的结果累积到sum中
		sum += scores[0][i];
	}
	// 概率化后的output 减去 真实答案就是损失。
	for(int i = 0; i < _labels_size; i++) 
		loutput[0][i] = scores[0][i] / sum - answer[i];
	// 释放scores内存
	FreeSpace(&scores);
}

// output中哪一维的数字最大 其对应的下标就是预测出的答案
int Classifier::predictByOutput(Tensor<cpu,2,double> output)
{
	// 初始化记录的结果 假设第0维就是答案
	int my_answer_pos = 0;		
	// 从output中找出比0维对应的数字还大的x维
	for(int i = 1; i < _labels_size; i++)
		if(output[0][i] > output[0][my_answer_pos])
			my_answer_pos = i;
	// 返回最终找到的下标
	return my_answer_pos;	
}

void Classifier::predict(vector<Example> test_examples)
{
	// 统计专用类声明了一个对象
	Counter the_counter;
	// 遍历所有参与预测的句子
	for(int i = 0; i < test_examples.size(); i++)
	{
		// output存储y'=w*x计算的结果
		Tensor<cpu,2,double> output;
		// output分配内存
		output = NewTensor<cpu>(Shape2(1,_labels_size),0.0);	
		// y' = w*x 这里的output就是y'。
		_layer.computeForwardScore(test_examples[i]._d_features,output);
		// 根据计算结果查找具体分成哪一类
		int result_pos = predictByOutput(output);
		// 统计预测正确的个数
		if(test_examples[i]._d_label[result_pos] == 1)
			the_counter.right_count++;
		// 统计预测个数
		the_counter.all_count++;
		// output释放内存
		FreeSpace(&output);
	}
	//  屏幕输出统计预测个数 和 预测正确个数
	cout << the_counter.right_count << "," << the_counter.all_count << endl;
	// 屏幕输出统计预测正确率
	cout << the_counter.rightRate() * 100 << "%" << endl;
}

Classifier::~Classifier()
{
	ShutdownTensorEngine<cpu>(); // 关闭矩阵运算
}
