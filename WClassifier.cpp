#include "WClassifier.h"
#include "Counter.h"

#include <limits.h>

WClassifier::WClassifier(int labels_size,NRMat<double> &word_emb)
{
	//打开矩阵运算
	InitTensorEngine<cpu>();
	//给标签表赋值
	this -> _labels_size = labels_size;
	// 获得词向量维度
	this -> _dim_size = word_emb.ncols();
	// 获得词向量个数
	int v_size = word_emb.nrows();
	// 为 _words 开辟内存	
	_words = NewTensor<cpu>(Shape2(v_size,_dim_size),0.0);
	// 复制入_words
	assign(_words,word_emb);
	// 初始化函数
	_layer.initial(_labels_size,3 * _dim_size);
}

void WClassifier::myTrain(vector<Example> train_examples)
{
	// 存储随机下标
	vector<int> random_pos;
	// 按顺序插入下标
	for(int i = 0; i< train_examples.size(); i++)
		random_pos.push_back(i);
	// 打乱下标
	random_shuffle(random_pos.begin(),random_pos.end());
	// 遍历每一个训练的句子
	for(int i = 0; i < train_examples.size(); i++)			
	{
		// 获得随机的句子下标
		int pos = random_pos[i];
		// 获得该句子的特征编码
		vector<int> &features = train_examples[pos]._d_features;
		// 获得该句子标签的编码 
		vector<int> &label = train_examples[pos]._d_label;
		// 获得句子的词语个数
		int words_num = features.size();
		// 为存句子所有词向量words_prime 申请内存 
		Tensor<cpu,2,double> words_prime = NewTensor<cpu>(Shape2(words_num,_dim_size),0.0);
		// words_prime 平均 为其申请内存
		Tensor<cpu,2,double> avg = NewTensor<cpu>(Shape2(1,_dim_size),0.0);
		// min 每维上的最小 为其申请内存
		Tensor<cpu,2,double> min = NewTensor<cpu>(Shape2(1,_dim_size),0.0);
		// max 每维上的最大  为其申请内存
		Tensor<cpu,2,double> max = NewTensor<cpu>(Shape2(1,_dim_size),0.0);
		// min,av,max 连接成一个向量 pool_merge  ,为其申请内存
		Tensor<cpu,2,double> pool_merge = NewTensor<cpu>(Shape2(1,3 * _dim_size),0.0);
		// 计算结果output  为其申请内存
		Tensor<cpu,2,double> output = NewTensor<cpu>(Shape2(1,_labels_size),0.0);
		// output的损失 为其申请内存
		Tensor<cpu,2,double> outputLoss = NewTensor<cpu>(Shape2(1,_labels_size),0.0);
		// 遍历这个句子的每一个特征
		for(int j = 0; j < features.size(); j++)
		{
			// 获得某特征的编码
			int feature_id = features[j];
			// 找到其词向量
			words_prime[j] += _words[feature_id];
		}
		// 求出avg
		avgWordEmb(words_prime,avg);
		// 求出 min
		minWordEmb(words_prime,min);
		// 求出 max
		maxWordEmb(words_prime,max);
		// 把min avg max 连接起来
		concat(min,avg,max,pool_merge);
		// y'=Wx
		_layer.computeForwardScore(pool_merge,output);
		// 计算y'的损失
		softmaxLoss(output,label,outputLoss);
		// 累积y'的损失
		_layer.computeBackwardLoss(pool_merge,outputLoss);
		// 更新W
		_layer.updateW();
		// 释放words_prime内存
		FreeSpace(&words_prime);
		// 释放avg内存
		FreeSpace(&avg);
		// 释放min内存
		FreeSpace(&min);
		// 释放max内存
		FreeSpace(&max);
		// 释放pool_merge内存
		FreeSpace(&pool_merge);
		// 释放output内存
		FreeSpace(&output);
		// 释放outputLoss内存
		FreeSpace(&outputLoss);
	}
}

void WClassifier::avgWordEmb(Tensor<cpu,2,double> words_prime,Tensor<cpu,2,double> avg)
{
	// 初始化avg
	avg = 0.0;
	// 获得词的个数
	int words_num = words_prime.size(0);
	// 遍历这个句子的所有词向量
	for(int i = 0; i < words_num; i++)
		avg[0] += words_prime[i]; // 累加
	// 求平均
	avg /= words_num;
}

void WClassifier::maxWordEmb(Tensor<cpu,2,double> words_prime,Tensor<cpu,2,double> max)
{
	// 初始化 max
	max = 0.0;
	// 获得词向量的个数
	int words_num = words_prime.size(0);
	// 遍历词向量的每一维
	for(int i = 0; i < _dim_size; i++)
	{
		// 初始化该维最大的数字
		double max_value = INT_MIN;
		// 遍历每个词向量这一维的数字
		for(int j = 0; j < words_num; j++)	
		{
			// 获得该维上的数字
			double cur_value = words_prime[j][i];
			// 是否超过当前最大
			if(max_value < cur_value)
				max_value = cur_value; // 覆盖之
		}
		// 存入max
		max[0][i] += max_value; 
	}
}

void WClassifier::minWordEmb(Tensor<cpu,2,double> words_prime,Tensor<cpu,2,double> min)
{
	// 初始化 min
	min = 0.0;
	// 获得词向量的个数
	int words_num = words_prime.size(0);
	// 遍历词向量的每一维
	for(int i = 0; i < _dim_size; i++)
	{
		// 初始化该维最小的数字
		double min_value = INT_MAX;
		// 遍历每个词向量这一维上的数字
		for(int j = 0; j < words_num; j++)	
		{
			// 获得该维上的数字
			double cur_value = words_prime[j][i];
			// 是否比当前的小
			if(min_value > cur_value)
				min_value = cur_value; // 覆盖之
		}
		// 存入min
		min[0][i] += min_value;
	}
}

void WClassifier::concat(Tensor<cpu,2,double> min,Tensor<cpu,2,double> avg,Tensor<cpu,2,double> max,Tensor<cpu,2,double> pool_merge)
{
	// 初始化pool_merge
	pool_merge = 0.0;
	// 获得向量维数
	int min_dim_size = min.size(1);
	// min加入pool_merge
	for(int i = 0; i < min_dim_size; i++)
		pool_merge[0][i] += min[0][i];
	//  获得向量维数
	int avg_dim_size = avg.size(1);
	// avg加入pool_merge 
	for(int i = min_dim_size; i < avg_dim_size + min_dim_size; i++)
		pool_merge[0][i] += avg[0][i];
	// 获得向量维数
	int max_dim_size = max.size(1);
	// max 加入pool_merge
	for(int i = min_dim_size + avg_dim_size; i < avg_dim_size + min_dim_size + max_dim_size; i++)
		pool_merge[0][i] += max[0][i];
}

void WClassifier::predict(vector<Example> test_examples)
{
	// 统计专用类声明了一个对象
	Counter the_counter;
	// 遍历所有参与预测的句子
	for(int i = 0; i < test_examples.size(); i++)
	{
		// 获得句子特征
		vector<int> &features = test_examples[i]._d_features;
		// 句子特征的数量
		int words_num = features.size();
		// 存储词向量
		Tensor<cpu,2,double> words_prime = NewTensor<cpu>(Shape2(words_num,_dim_size),0.0);
		// 存储句子词向量求和
		Tensor<cpu,2,double> avg = NewTensor<cpu>(Shape2(1,_dim_size),0.0);
		// min 
		Tensor<cpu,2,double> min = NewTensor<cpu>(Shape2(1,_dim_size),0.0);
		// max 
		Tensor<cpu,2,double> max = NewTensor<cpu>(Shape2(1,_dim_size),0.0);
		// 三个向量连接
		Tensor<cpu,2,double> pool_merge = NewTensor<cpu>(Shape2(1,3 * _dim_size),0.0);
		// output存储y'=w*x计算的结果
		Tensor<cpu,2,double> output;
		// output分配内存
		output = NewTensor<cpu>(Shape2(1,_labels_size),0.0);	
		// 遍历这个句子的特征
		for(int j = 0; j < features.size(); j++)
		{
			// 获得特征下标编码
			int feature_id = features[j];
			// 词向量都存入words_prime 
			words_prime[j] += _words[feature_id];
		}
		// 所有词向量求平均
		avgWordEmb(words_prime,avg);
		// 每个维度取最小
		minWordEmb(words_prime,min);
		// 每个维度取最大
		maxWordEmb(words_prime,max);
		// 连成一个词向量	
		concat(min,avg,max,pool_merge);
		// y' = w*x 这里的output就是y'。
		_layer.computeForwardScore(pool_merge,output);
		// 根据计算结果查找具体分成哪一类
		int result_pos = predictByOutput(output);
		// 统计预测正确的个数
		if(test_examples[i]._d_label[result_pos] == 1)
			the_counter.right_count++;
		// 统计预测个数
		the_counter.all_count++;
		// output释放内存
		FreeSpace(&output);
		// words_prime释放内存
		FreeSpace(&words_prime);
		// avg 释放内存
		FreeSpace(&avg);
		// min 释放内存
		FreeSpace(&min);
		// max 释放内存
		FreeSpace(&max);
		// pool_merge 释放内存
		FreeSpace(&pool_merge);
	}
	//  屏幕输出统计预测个数 和 预测正确个数
	cout << the_counter.right_count << "," << the_counter.all_count << endl;
	// 屏幕输出统计预测正确率
	cout << the_counter.rightRate() * 100 << "%" << endl;
}

// output中哪一维的数字最大 其对应的下标就是预测出的答案
int WClassifier::predictByOutput(Tensor<cpu,2,double> output)
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


void WClassifier::softmaxLoss(Tensor<cpu,2,double> output,const vector<int> &answer,Tensor<cpu,2,double> loutput)
{
	// 初始化最大分量的下标
	int my_answer_pos = 0;		
	// 遍历每维找最大的分量
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

WClassifier::~WClassifier()
{
	// 释放_words内存
	FreeSpace(&_words);
	// 关闭矩阵运算库
	ShutdownTensorEngine<cpu>();
}
