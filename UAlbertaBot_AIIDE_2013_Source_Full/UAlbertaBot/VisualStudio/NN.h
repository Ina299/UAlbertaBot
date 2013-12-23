#pragma once
class NN
{
public:
	NN();
	~NN();
};

public class NeuralNetwork {
	private Layer inputLayer;   // ���͑w
	private Layer hiddenLayer;  // �B��w
	private Layer outputLayer;  // �o�͑w

	/**
	* �j���[�����l�b�g��������
	* @param numInputNodes ���͑w�̃m�[�h��
	* @param numHiddenNodes �B��w�̃m�[�h��
	* @param numOutputNodes �o�͑w�̃m�[�h��
	*/
	public void init(int numInputNodes, int numHiddenNodes, int numOutputNodes) {
		inputLayer = new Layer();
		hiddenLayer = new Layer();
		outputLayer = new Layer();

		// ���͑w�i�e�w���Ȃ����Ƃɒ��Ӂj
		inputLayer.numNodes = numInputNodes;
		inputLayer.numChildNodes = numHiddenNodes;
		inputLayer.numParentNodes = 0;
		inputLayer.init(numInputNodes, null, hiddenLayer);
		inputLayer.randomizeWeights();

		// �B��w
		hiddenLayer.numNodes = numHiddenNodes;
		hiddenLayer.numChildNodes = numOutputNodes;
		hiddenLayer.numParentNodes = numInputNodes;
		hiddenLayer.init(numHiddenNodes, inputLayer, outputLayer);
		hiddenLayer.randomizeWeights();

		// �o�͑w�i�q�w���Ȃ����Ƃɒ��Ӂj
		outputLayer.numNodes = numOutputNodes;
		outputLayer.numChildNodes = 0;
		outputLayer.numParentNodes = numHiddenNodes;
		outputLayer.init(numOutputNodes, hiddenLayer, null);
		// �o�͑w�ɏd�݂͂Ȃ��̂�randomizeWeights()�͕K�v�Ȃ�
	}


}
