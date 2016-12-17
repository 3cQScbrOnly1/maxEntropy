#include "MyAlphabet.h"
#include "FeatureExtractor.h"

// 神马也不做
MyAlphabet::MyAlphabet()
{}

MyAlphabet::MyAlphabet(vector<Instance> the_instances)
{
	// 遍历所有参与建立特征表 标签表的句子
	for(int i = 0; i < the_instances.size(); i++)		
	{
		// 获得一个句子的标签
		string &label = the_instances[i]._label;
		// 标签表里是否存在的标志位
		bool LABEL_EXIST = false;
		// 遍历标签表，查找标签是否存在
		for(int j = 0; j < _labels_alphabet.size(); j++)
			if(label == _labels_alphabet[j])
				LABEL_EXIST = true;
		// 将新发现的标签加入标签表
		if(!LABEL_EXIST)
			_labels_alphabet.push_back(label);
		// 获得句子
		vector<string> &words = the_instances[i]._sentence;
		// 存储句子特征
		vector<string> features;
		// 特征抽取类
		FeatureExtractor the_extractor;
		// 抽取特征
		the_extractor.extract(words,features);
		// 用于查找特征
		map<string,int>::iterator it;
		// 遍历特征表
		for(int j = 0; j < features.size(); j++)
		{
			// 查找特征是否在特征表里存在
			it = _features_alphabet.find(features[j]);
			if(it == _features_alphabet.end())
			{
				int pos = _features_alphabet.size(); // 以出现顺序作为序号
				_features_alphabet.insert(make_pair(features[j],pos));	// 加入新的特征
			}
		}
	}
		
	// 屏幕输出标签和特征个数
	cout << "label size: " << _labels_alphabet.size() << endl;
	cout << "features size: " << _features_alphabet.size() << endl;
}
