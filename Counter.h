class Counter
{
	public:
		// 所有个数
		int all_count;
		// 正确个数
		int right_count;
		// 默认构造函数
		Counter();
		// 重置
		void reset();
		// 计算正确率
		double rightRate();
};
