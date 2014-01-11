#include "Common.h"
#include "Neural.h"


// constructor
Neural::Neural()
:selfRace(BWAPI::Broodwar->self()->getRace())
, enemyRace(BWAPI::Broodwar->enemy()->getRace())
{
	count = 0;
	//2�i���ŕ\�����߂�2�{,�G���j�b�g�������j�b�g�ŋ�ʂ���
	//+2�͎푰
	num_states = setNumState()*unit_count*2 + 2;
	num_input = num_actions + num_states;
	setStates();
	setActions();
	createNetwork();
}

// get an instance of this
Neural & Neural::Instance()
{
	static Neural instance;
	return instance;
}


void Neural::onEnd(const bool isWinner)
{
	// if the game ended before the tournament time limit
	// ��V�ɂ����outputs���X�V����
	if (BWAPI::Broodwar->getFrameCount() < Options::Tournament::GAME_END_FRAME)
	{
		if (isWinner)
		{
			//��V�̐ݒ�
			outputs[count][0] = 1.0;
			for (float i = count-1; i > 0; --i){
				outputs[i][0] = outputs[i][0]
					+alpha*(1.0+gamma*outputs[i+1][0]-outputs[i][0]);
			}
		}
		else
		{
			//��V�̐ݒ�
			outputs[count][0] = -1.0;
			for (float i = count - 1; i > 0; --i){
				outputs[i][0] = outputs[i][0]
					+ alpha*(-1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
			}
		}
	}
	// otherwise game timed out so use in-game score
	// ���菟��
	else
	{
		if (getScore(BWAPI::Broodwar->self()) > getScore(BWAPI::Broodwar->enemy()))
		{
			outputs[count][0] = 1.0;
			for (float i = count - 1; i > 0; --i){
				outputs[i][0] = outputs[i][0]
					+ alpha*(1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
			}
		}
		else
		{
			outputs[count][0] = -1.0;
			for (float i = count - 1; i > 0; --i){
				outputs[i][0] = outputs[i][0]
					+ alpha*(-1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
			}
		}


	FANN::training_data data;


	data.set_train_data(count,num_input,(float**)&inputs[0],
									num_output,(float**)&outputs[0]);
	// Initialize and train the network with the data
	net.init_weights(data);
	net.train_on_data(data, max_iterations,
		iterations_between_reports, desired_error);

	net.save(writeDir + BWAPI::Broodwar->enemy()->getName() + ".net");
	}
}

// Test function that demonstrates usage of the fann C++ wrapper
void	Neural::createNetwork()
{
	//	cout << endl << "Creating network." << endl;
	if (net.create_from_file(readDir + BWAPI::Broodwar->enemy()->getName() + ".net")){

	}
	else
	{
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
	count++;
	srand((unsigned int)time(NULL));
	//�C�v�V����=0.2�Ń����_����
	if (rand()%10>8){
		for (int j; j < num_actions; ++j){
			int flag = ((rand()%(int)pow(2.0, num_actions)) >> j) % 2;
			flag == 1 ? inputs[count][j] = 1.0 : inputs[count][j] = 0.0;
		}
	}
	else{
		selectBestAction();
		inputs[count] = bestinput;
	}


}

void Neural::selectBestAction(){

	fann_type *best_out;
	std::vector<float> 	actions(num_actions,0.0);
	std::vector<float>	states = getState();
	std::vector<float>	input;
	//2��num_actions��ɂ��đ�����
	for (int i; i < (int)pow(2.0,num_actions);++i){
		for (int j; j < num_actions; ++j){
			int flag = (i >> j) % 2;
			flag == 1 ? actions[j] = 1.0 : actions[j] = 0.0;
		}
			input.insert(actions.end(), states.begin(),
				states.end());
			fann_type *calc_out = net.run(&input[0]);
			if (best_out[0] < calc_out[0]){
				best_out[0] = calc_out[0];
				bestinput = input;
				outputs[count][0] = best_out[0];
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
	return BWAPI::Broodwar->getFrameCount() % 48 == 0;
}

void Neural::setStates(){
	//�S�Ẵ��j�b�g�ɂ���Region�̐�����RegionID��~��
	std::map<int, int>	own_region_num;
	std::map<int, int>	opponent_region_num;
	BOOST_FOREACH(BWAPI::Region * currentRegion, BWAPI::Broodwar->getAllRegions()){
		//		region_num[currentRegion->BWAPI::Region::getRegionGroupID()]=0;
		own_region_num.insert(std::map<int, int>::value_type(currentRegion->BWAPI::Region::getRegionGroupID(), 0));
		opponent_region_num.insert(std::map<int, int>::value_type(currentRegion->BWAPI::Region::getRegionGroupID(), 0));
	}
	//�S�Ẵ��j�b�g��Region���𑫂�
	BOOST_FOREACH(BWAPI::Unit * currentUnit, BWAPI::Broodwar->enemy()->getUnits()){
		opponent_region_num[currentUnit->BWAPI::Unit::getRegion()->BWAPI::Region::getRegionGroupID()] ++;
	}
	BOOST_FOREACH(BWAPI::Unit * currentUnit, BWAPI::Broodwar->self()->getUnits()){
		own_region_num[currentUnit->BWAPI::Unit::getRegion()->BWAPI::Region::getRegionGroupID()] ++;
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

std::vector<float>	& Neural::getActions(){
	return inputs[count];
}