#include "InstanceReader.h"
#include "MyAlphabet.h"
#include "InstanceToExampleTransformer.h"
#include "Classifier.h"
#include "WClassifier.h"
#include "WordEmbReader.h"

#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

int main(int argc , char *argv[])
{
	// 声明一个存储训练集的对象
	vector<Instance> train_instances;
	// 声明一个存储测试集的对象
	vector<Instance> test_instances;
	// 声明一个读取文件类对象
	InstanceReader the_reader;
	// argv[1]是训练集文件路径 读到的内容存入train_instances
	the_reader.load(argv[1],train_instances);
	// argv[2]是测试集文件路径 读到的内容存入 test_instances
	the_reader.load(argv[2],test_instances);
	// 声明一个存储标签表和特征表的对象，按训练集构造标签表和特征表
	MyAlphabet the_alphabet(train_instances);
	// 声明一个转化对象，转化的规则在标签表和特征表中
	InstanceToExampleTransformer i2e_transformer;
	// 声明一个存储转化后的训练集对象
	vector<Example> train_examples;
	// 声明一个存储转化后的测试集集对象
	vector<Example> test_examples;
	// 开始按照标签表和特征表将训练集中的所有句子进行转化，结果存入train_examples
	i2e_transformer.transfer(the_alphabet,train_instances,train_examples);
	// 开始按照标签表和特征表将测试集中的所有句子进行转化，结果存入 test_examples
	i2e_transformer.transfer(the_alphabet,test_instances,test_examples);
	// 声明一个词向量读取对象
	WordEmbReader the_w_reader;
	// 临时存放词向量
	NRMat<double> word_emb;
	// 读取词向量
	the_w_reader.readWordEmb(word_emb,the_alphabet._features_alphabet,argv[3]);
	// 声明一个分类器对象
	WClassifier the_w_classifier(the_alphabet._labels_alphabet.size(),word_emb);
	// 声明一个类型转化对象
	stringstream chars_to_int;
	// 最大迭代次数
	int max_loop;
	// 最大迭代次数是否设置
	if( argc < 5 )
		max_loop = 10; // 默认10
	else
	{
		// argv[3]进行类型转化
		chars_to_int << argv[4];
		chars_to_int >> max_loop;
	}
	// 屏幕输出max_loop
	cout << "max_loop" << max_loop << endl;
	for (int i = 0; i < max_loop; i++)
	{
		// 训练参数
		the_w_classifier.myTrain(train_examples);
		// 预测 测试性能
		the_w_classifier.predict(test_examples);
	}
	// 程序结束	
	return 0;
}
