#include "Counter.h"

Counter::Counter()
{
	// 成员变量初始化成0
	all_count = 0;
	right_count = 0;
}

void Counter::reset()
{
	// 重置之
	all_count = 0;
	right_count = 0;
}

double Counter::rightRate()
{
	// 返回正确率
	return (double)right_count / all_count;
}
