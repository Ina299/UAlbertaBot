#pragma once
class Layer
{
	Layer();
	~Layer();

public:
	static	Layer &	Instance();
	int numNodes; // �m�[�h��
	int numChildNodes; // �q�w�̃m�[�h��
	int numParentNodes; // �e�w�̃m�[�h��
	/*
	double[][] weights; // ���̑w�Ǝq�w�Ԃ̏d��
	double[] neuronValues; // �m�[�h�̊����l
	double[] teacherValues; // ���t�M��
	double[] errors; // �덷
	double[] biasWeights; // �o�C�A�X�̏d��
	double[] biasValues; // �o�C�A�X�l�i�o�C�A�X�̏d��*�o�C�A�X�l��������臒l�j
	double learningRate; // �w�K��

	Layer parentLayer; // �e�w�ւ̎Q��
	Layer childLayer; // �q�w�ւ̎Q��

	Random rand;
	*/
};
