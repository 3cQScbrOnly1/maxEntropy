#ifndef _MYALPHABET_H_
#define _MYALPHABET_H_

#include "Instance.h"

#include <map>
#include <vector>

using namespace std;

class MyAlphabet
{
	public:
		// 特征表 
		map<string,int> _features_alphabet;
		// 标签表 
		vector<string>  _labels_alphabet;
		// 默认构造函数 不调用
		MyAlphabet();
		// 用训练集构造
		MyAlphabet(vector<Instance> the_instances);
};

#endif
