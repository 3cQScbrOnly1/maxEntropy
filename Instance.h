#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <vector>
#include <iostream>

using namespace std;

class Instance 
{
	public:
		// 句子标签 (哪一类)
		string _label;
		// 句子
		vector<string> _sentence;
};

#endif
