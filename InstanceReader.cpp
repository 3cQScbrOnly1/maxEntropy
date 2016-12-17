#include <fstream>
#include <boost/algorithm/string.hpp>

#include "InstanceReader.h"

using namespace std;
using namespace boost;

InstanceReader::InstanceReader()
{}

void InstanceReader::load(char *path,vector<Instance> &the_instances)
{
	//path 文件路径,打开文件流 
	ifstream file(path);
	// 按行读取数据
	string line;
	while(getline(file,line))
	{
		// the_ins 临时存储
		Instance the_ins;
		// 存储line 按空格切分的结果
		vector<string> words;
		// 对line按照空格切分
		split(words,line,is_any_of(" "));
		// 存储label
		the_ins._label = words[0]; 
		// 存储句子
		for(int i = 1; i < words.size(); i++)
			if(words[i] != "") // 去除无效字符
				the_ins._sentence.push_back(words[i]); // 句子存入临时存储
		the_instances.push_back(the_ins); // 保存临时存储
	}
	//关闭文件流
	file.close();
}
