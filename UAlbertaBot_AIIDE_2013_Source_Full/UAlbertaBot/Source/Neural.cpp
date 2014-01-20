#include "Common.h"
#include "Neural.h"
#include <ios>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>

//��������j���[�����l�b�g�̃p�����[�^
const int num_actions = 4;
const int unit_count = 2;
const int num_layers = 3;
const int num_hidden = 3;
const int num_output = 1;
const float learning_rate = 0.7f;
//origin
//const float desired_error = 0.001f;
//const int max_iterations = 300000;
//const int iterations_between_reports = 1000;
const float desired_error = 0.01f;
const int max_iterations = 1;
const int iterations_between_reports = 0;
//�������狭���w�K�̃p�����[�^
const float gamma = 0.95;
const float alpha = 0.7;
const std::string writeDir = "bwapi-data\\write\\";
const std::string readDir = "bwapi-data\\read\\";

// constructor
Neural::Neural()
:selfRace(BWAPI::Broodwar->self()->getRace())
, enemyRace(BWAPI::Broodwar->enemy()->getRace())
{
	BWAPI::Broodwar->sendText("Neural constructed");
	
	//2�i���ŕ\�����߂�2�{,�G���j�b�g�������j�b�g�ŋ�ʂ���
	//+2�͎푰
	num_states = setNumState()*unit_count*2 + 2;
	num_input = num_actions + num_states;
	createNetwork();
	setStates();
	setActions();
}

// get an instance of this
Neural & Neural::Instance()
{
	static Neural instance;
	return instance;
}


void Neural::onEnd(const bool isWinner)
{
	size_t count = outputs.size();
	/*
	std::stringstream ss;
	ss << count;
	std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() + ss.str() + ".txt";
	std::ofstream f_out(writeFile.c_str());
	f_out << "test";
	f_out.close();
	*/
	// if the game ended before the tournament time limit
	// ��V�ɂ����outputs���X�V����
	if (BWAPI::Broodwar->getFrameCount() < Options::Tournament::GAME_END_FRAME)
	{
		if (isWinner)
		{
			//��V�̐ݒ�
			outputs[count-1][0] = 1.0;
			
			for (size_t i = count - 2 ; i > 0 ; --i){
				float temp = outputs[i][0];
				float neko;
				//test
				neko = gamma * outputs[i + 1][0];
				temp += alpha * (1.0 + neko - temp);
				std::vector<float> x;
				x.push_back(temp);
				outputs[i] = x;	
			}
		}
		else
		{
			//��V�̐ݒ�
			/*	
			std::vector<float> lose(1, -1.0);
			outputs[count]=lose;
			*/
			outputs[count-1][0] = -1.0;
			for (size_t i = count - 2; i > 0; --i){
				/*
				std::cout << "testa" << std::endl;
				outputs[i][0] = outputs[i][0]
					+ alpha*(-1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
					*/
				float temp = outputs[i][0];
				float neko;
				//test
				neko = gamma * outputs[i + 1][0];
				temp += alpha * (-1.0 + neko - temp);
				std::vector<float> x;
				x.push_back(temp);
				outputs[i] = x;	
			}
		}
	}
	// otherwise game timed out so use in-game score
	// ���菟��
	else
	{
		if (getScore(BWAPI::Broodwar->self()) > getScore(BWAPI::Broodwar->enemy()))
		{
			outputs[count-1][0] = 1.0;
			for (size_t i = count - 2; i > 0; --i){
				/*
				std::cout << "testa" << std::endl;
				outputs[i][0] = outputs[i][0]
					+ alpha*(1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
					*/
				float temp = outputs[i][0];
				float neko;
				//test
				neko = gamma * outputs[i + 1][0];
				temp += alpha * (1.0 + neko - temp);
				std::vector<float> x;
				x.push_back(temp);
				outputs[i] = x;
			}
			
		}
		else
		{
			outputs[count-1][0] = -1.0;
			for (size_t i = count - 2; i > 0; --i){
				/*
				std::cout << "testa" << std::endl;
				outputs[i][0] = outputs[i][0]
					+ alpha*(-1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
				*/
				float temp = outputs[i][0];
				float neko;
				//test
				neko = gamma * outputs[i + 1][0];
				temp += alpha * (-1.0 + neko - temp);
				std::vector<float> x;
				x.push_back(temp);
				outputs[i] = x;
			}
			
		}
	}

	FANN::training_data data;
//	float **tem1 = new float*[count];
//	float **tem2 = new float*[count];
//	boost::shared_ptr<float **> p(&inputs[0]);
//	boost::shared_ptr<float **> q(&inputs[0]);
	//		tem1[i] = new float[inputs[i].size()];
/*	for (int i = 0; i < count; ++i)
	{
		tem1[i] = new float[inputs[i].size()];
		tem2[i] = new float[outputs[i].size()];
		tem1[i] = &inputs[i][0];
		tem2[i] = &outputs[i][0];
	}
	*/
//	data.set_train_data(count-1,num_input,tem1,
	//								num_output,tem2);
	data.set_train_data(inputs,outputs);
	BWAPI::Broodwar->printf("Create Data");
	/*
	std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() +"DataCreate" +".txt";
	std::ofstream p_out(writeFile.c_str());
	p_out << "failed";
	p_out.close();
	*/
	// Initialize and train the network with the data
	net.init_weights(data);
	net.train_on_data(data, max_iterations,
		iterations_between_reports, desired_error);
//	BWAPI::Broodwar->printf("Successfully trained");

	if (!net.save(writeDir + "test" + ".net"))
	{
		std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() + ".txt";
		std::ofstream f_out(writeFile.c_str());
		f_out << "failed";
		f_out.close();
	}
	std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() + "for_omae.txt";
	std::ofstream f_out(writeFile.c_str());
	f_out << "failed";
	f_out.close();
	/*
	for (int i = 0; i < count; ++i)
	{
		std::stringstream ss;
		ss << i;
		std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() + ss.str() + ".txt";
		std::ofstream f_out(writeFile.c_str());
		f_out << "failed";
		f_out.close();
		//std::vector<float>().swap(inputs[i]);
		//std::vector<float>().swap(outputs[i]);
		delete[] tem1[i];
		delete[] tem2[i];
		inputs[i][0] = 0;
		outputs[i][0] = 0;
	}
	*/
	/*
	for (int i = 0; i < count; ++i)
	{
		std::stringstream ss;
		ss << i;
		std::string priteFile = writeDir + BWAPI::Broodwar->enemy()->getName() + ss.str() + "_delete.txt";
		std::ofstream p_out(priteFile.c_str());
		p_out << "failed";
		p_out.close();
		//std::vector<float>().swap(inputs[i]);
		//std::vector<float>().swap(outputs[i]);
				delete[] tem1[i];
				delete[] tem2[i];
	//	inputs[i][0] = 0;
	//	outputs[i][0] = 0;
	}
	*/

//	delete[] tem1;
//	delete[] tem2;
//	inputs.clear();
//	outputs.clear();
//	std::vector<std::vector<float> >().swap(inputs);
//	std::vector<std::vector<float> >().swap(outputs);
	/*
	delete[]	tem1;
	delete[]	tem2;
	*/
	std::string sriteFile = writeDir + BWAPI::Broodwar->enemy()->getName() + "_finish.txt";
	std::ofstream l_out(sriteFile.c_str());
	l_out << "failed";
	l_out.close();
}

// Test function that demonstrates usage of the fann C++ wrapper
void	Neural::createNetwork()
{
	//	cout << endl << "Creating network." << endl;
	//BWAPI::Broodwar->enemy()->getName()
	if (net.create_from_file(writeDir + "test" + ".net")){
		BWAPI::Broodwar->sendText("network read");
	}
	else
	{
		BWAPI::Broodwar->sendText("new network constructed");
		net.create_standard(num_layers, num_input, num_hidden, num_output);

		net.set_learning_rate(learning_rate);

		net.set_activation_steepness_hidden(1.0);
		net.set_activation_steepness_output(1.0);
		//change if you need
		net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
		net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

		// Set additional properties such as the training algorithm
		net.set_training_algorithm(FANN::TRAIN_QUICKPROP);
	}
}


const int Neural::getScore(BWAPI::Player * player) const
{
	return player->getBuildingScore() + player->getKillScore() + player->getRazingScore() + player->getUnitScore();
}

void Neural::setActions()
{
//	BWAPI::Broodwar->sendText("SelectStrategy");
	srand((unsigned int)time(NULL));
	//�C�v�V����=0.1�Ń����_����
	if (rand()%10>8){
		std::vector<float> 	actions(num_actions, 0.0);
		std::vector<float>	input;
		std::vector<float>  best_out(1,-10.0);
		//2��num_actions��ɂ��đ�����
			for (int j=0; j < num_actions; ++j){
				int flag = ((rand() % (int)pow(2.0, num_actions)) >> j) % 2;
				flag == 1 ? actions[j] = 1.0 : actions[j] = 0.0;
			}
			input.insert(input.end(), actions.begin(),
				actions.end());
			input.insert(input.end(), states.begin(),
				states.end());
			best_out[0]=*net.run(&input[0]);
			outputs.push_back(best_out);
			inputs.push_back(input);
	}
	else{
		selectBestAction();
	}
}

void Neural::selectBestAction(){

	std::vector<float>  best_out(1,-10.0);
	std::vector<float> 	actions(num_actions,0.0);
	std::vector<float>	input;
	//2��num_actions��ɂ��đ�����
	for (int i=0; i < (int)pow(2.0,num_actions);++i){
		for (int j=0; j < num_actions; ++j){
			int flag = (i >> j) % 2;
			flag == 1 ? actions[j] = 1.0 : actions[j] = 0.0;
		}
		input.insert(input.end(), actions.begin(),
			actions.end());
			input.insert(input.end(), states.begin(),
				states.end());
			//test
			float calc_out = *net.run(&input[0]);
			if (best_out[0] < calc_out){
				best_out[0] = calc_out;
				outputs.push_back(best_out);
				inputs.push_back(input);
		}
	}
}


void Neural::update(){
	if (neuralUpdateFrame()){
		setStates();
		setActions();
	}
}

bool Neural::neuralUpdateFrame()
{
	return BWAPI::Broodwar->getFrameCount() % 120 == 0;
}

void Neural::setStates(){
	//�S�Ẵ��j�b�g�ɂ���Region�̐�����RegionID��~��
	std::map<int, int>	own_region_num;
	std::map<int, int>	opponent_region_num;
	BOOST_FOREACH(BWAPI::Region * currentRegion, BWAPI::Broodwar->getAllRegions()){
		//		region_num[currentRegion->BWAPI::Region::getRegionGroupID()]=0;
		own_region_num.insert(std::map<int, int>::value_type(currentRegion->getRegionGroupID(), 0));
		opponent_region_num.insert(std::map<int, int>::value_type(currentRegion->getRegionGroupID(), 0));
	}
	//�S�Ẵ��j�b�g��Region���𑫂�
	BOOST_FOREACH(BWAPI::Unit * currentUnit, BWAPI::Broodwar->enemy()->getUnits()){
		opponent_region_num[currentUnit->getRegion()->getRegionGroupID()] ++;
	}
	BOOST_FOREACH(BWAPI::Unit * currentUnit, BWAPI::Broodwar->self()->getUnits()){
		own_region_num[currentUnit->getRegion()->getRegionGroupID()] ++;
	}
	//�eRegion���Ƃɖ������j�b�g����1�A2�A3�A4�ȏ�ŏꍇ�킯
	std::vector<float>	temp(num_states, 0.0);
	int count_temp = 0;
	int value;
	//map��foreach���܂킵��region���Ƃ�Unit����vector�ɕϊ�
	BOOST_FOREACH(boost::tie(boost::tuples::ignore, value), own_region_num){
		if (value == 0){
			temp[count_temp] = 0;
			temp[count_temp + 1] = 0;
		}
		else if (value == 1){
			temp[count_temp] = 0;
			temp[count_temp + 1] = 1;

		}
		else if (value == 2){
			temp[count_temp] = 1;
			temp[count_temp + 1] = 0;

		}
		else if (value > 2){
			temp[count_temp] = 1;
			temp[count_temp + 1] = 1;
		}

		count_temp += 2;
	}
	BOOST_FOREACH(boost::tie(boost::tuples::ignore, value), opponent_region_num){
		if (value == 0){
			temp[count_temp] = 0;
			temp[count_temp + 1] = 0;
		}
		else if (value == 1){
			temp[count_temp] = 0;
			temp[count_temp + 1] = 1;

		}
		else if (value == 2){
			temp[count_temp] = 1;
			temp[count_temp + 1] = 0;

		}
		else if (value > 2){
			temp[count_temp] = 1;
			temp[count_temp + 1] = 1;
		}

		count_temp += 2;
	}

	if (enemyRace == BWAPI::Races::Protoss)
	{
		temp[count_temp] = 0;
		temp[count_temp+1] = 0;
	}
	else if (enemyRace == BWAPI::Races::Terran)
	{
		temp[count_temp] = 0;
		temp[count_temp + 1] = 1;
	}
	else if (enemyRace == BWAPI::Races::Zerg)
	{
		temp[count_temp] = 1;
		temp[count_temp + 1] = 0;
	}
	else
	{
//		BWAPI::Broodwar->printf("Enemy Race Unknown");
		temp[count_temp] = 1;
		temp[count_temp + 1] = 1;
	}
	
	//���[�J���ϐ��̏�Ԃ������o�ϐ��ƃX���b�v
	states.swap(temp);

}

int Neural::setNumState(){
	//Region�̐���Ԃ�
	int region_count = 0;
	BOOST_FOREACH(BWAPI::Region * currentRegion, BWAPI::Broodwar->getAllRegions()){
		region_count++;
	}
	return region_count;
}

const std::vector<float> & Neural::getActions(){
	return inputs.back();
}