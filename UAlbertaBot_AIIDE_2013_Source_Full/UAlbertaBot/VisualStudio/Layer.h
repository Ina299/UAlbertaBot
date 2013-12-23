#pragma once
class Layer
{
	Layer();
	~Layer();

public:
	static	Layer &	Instance();
	int numNodes; // ノード数
	int numChildNodes; // 子層のノード数
	int numParentNodes; // 親層のノード数
	/*
	double[][] weights; // この層と子層間の重み
	double[] neuronValues; // ノードの活性値
	double[] teacherValues; // 教師信号
	double[] errors; // 誤差
	double[] biasWeights; // バイアスの重み
	double[] biasValues; // バイアス値（バイアスの重み*バイアス値がいわゆる閾値）
	double learningRate; // 学習率

	Layer parentLayer; // 親層への参照
	Layer childLayer; // 子層への参照

	Random rand;
	*/
};
