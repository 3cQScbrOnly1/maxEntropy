#include "InstanceToExampleTransformer.h"
#include "FeatureExtractor.h"


// 标签 特征 转数字 
void InstanceToExampleTransformer::transfer(MyAlphabet the_alphabet,vector<Instance> the_instances,vector<Example> &the_examples)
{
	// 遍历所有需要转化的句子
	for(int i = 0; i < the_instances.size(); i++)	
	{
		// 临时存储转化后句子的标签 特征
		Example the_example;
		// 用标签表转换标签
		labelTransfer(the_alphabet._labels_alphabet,the_instances[i]._label,the_example._d_label);
		// 存储句子特征
		vector<string> features;
		// 抽取特征类
		FeatureExtractor the_extractor;
		// 抽取句子特征
		the_extractor.extract(the_instances[i]._sentence,features);
		// 句子特征用特征表转数字
		featuresTransfer(the_alphabet._features_alphabet,features,the_example._d_features);
		// 存储转化结果
		the_examples.push_back(the_example);
	}
}

void InstanceToExampleTransformer::labelTransfer(vector<string> labels_alphabet,string label,vector<int> &d_label)
{
	// 遍历标签表 
	for(int i = 0; i < labels_alphabet.size(); i++)
		if(labels_alphabet[i] == label) //  是否是第i个分类结果
			d_label.push_back(1); // 是该类分类结果就在对应位上写1
		else
			d_label.push_back(0); // 不是该类分类结果就在对应位上写0
}

void InstanceToExampleTransformer::featuresTransfer(map<string,int> features_alphabet,vector<string> features,vector<int> &d_features)
{
	// 遍历特征表
	for(int i = 0; i < features.size(); i++)
	{
		map<string,int>::iterator it = features_alphabet.find(features[i]);
		// 记录特征在特征表中的对应的序号 (稀疏向量的处理)
		if(it != features_alphabet.end())
			d_features.push_back(it -> second);
	}
}
