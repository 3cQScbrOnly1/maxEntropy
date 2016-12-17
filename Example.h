#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <vector>

using namespace std;

class Example
{
	public:
		vector<int> _d_label; // 存储经过标签表转化后的句子标签
		vector<int> _d_features; // 存储经过特征表转化后的句子特征(特征表的下标)  
};

#endif
