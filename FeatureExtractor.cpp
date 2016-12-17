#include "FeatureExtractor.h"

void FeatureExtractor::extract(vector<string> words,vector<string> &features)
{
	features.clear();
	// 这里以句子中的所有词和连续的两个词语作为特征
	for(int i = 0; i < words.size(); i++)
	{
		// 抽取一个词语
		features.push_back(words[i]);
		// 抽取连续两个词语
		//if(i + 1 < words.size())
			//features.push_back(words[i] + "#" + words[i+1]);
	}
}
