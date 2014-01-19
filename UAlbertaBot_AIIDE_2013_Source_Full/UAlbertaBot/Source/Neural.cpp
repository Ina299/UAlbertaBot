#include "Common.h"
#include "Neural.h"


static FANN::neural_net net;

// constructor
Neural::Neural()
:selfRace(BWAPI::Broodwar->self()->getRace())
, enemyRace(BWAPI::Broodwar->enemy()->getRace())
{
	BWAPI::Broodwar->sendText("Neural constructed");
	gamma = 0.95;
	alpha = 0.7;
	learning_rate = 0.7f;
	desired_error = 0.001f;
	writeDir = "D:\\StarCraft\\bwapi-data\\write\\";
	readDir = "D:\\StarCraft\\bwapi-data\\read\\";
	//2進数で表すために2倍,敵ユニット味方ユニットで区別する
	//+2は種族
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
	// if the game ended before the tournament time limit
	// 報酬によってoutputsを更新する
	if (BWAPI::Broodwar->getFrameCount() < Options::Tournament::GAME_END_FRAME)
	{
		if (isWinner)
		{
			//報酬の設定
			outputs[count-1][0] = 1.0;
			/*
			std::vector<std::vector<float>>::iterator it = outputs.end();
			it=it-2;
			while (it != outputs.begin()){
				*it[0] += alpha*(1.0 + gamma*(*(it+1).back() - it.front());
				--it;
			}
			*/
			for (size_t i = count - 2 ; i > 0 ; --i){
				float temp = outputs[i][0];
				float neko;
				//test
				neko = gamma * outputs[i+1][0];
				temp += alpha * (-1.0 + neko - temp);
				std::vector<float> x;
				x.push_back(temp);
				outputs[i] = x;
			}
		}
		else
		{
			//報酬の設定
			/*	
			std::vector<float> lose(1, -1.0);
			outputs[count]=lose;
			*/
			outputs[count-1][0] = -1.0;
			for (float i = count - 2; i > 0; --i){
				outputs[i][0] = outputs[i][0]
					+ alpha*(-1.0 + gamma*outputs[i + 1][0] - outputs[i][0]);
			}
		}
	}
	// otherwise game timed out so use in-game score
	// 判定勝ち
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
	BWAPI::Broodwar->sendText("SelectStrategy");
	srand((unsigned int)time(NULL));
	//イプシロン=0.1でランダム化
	if (rand()%10>8){
		std::vector<float> 	actions(num_actions, 0.0);
		std::vector<float>	input;
		std::vector<float>  best_out(1,-10.0);
		//2のnum_actions乗について総当り
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
	//2のnum_actions乗について総当り
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
	//全てのユニットについてRegionの数だけRegionIDを蓄積
	std::map<int, int>	own_region_num;
	std::map<int, int>	opponent_region_num;
	BOOST_FOREACH(BWAPI::Region * currentRegion, BWAPI::Broodwar->getAllRegions()){
		//		region_num[currentRegion->BWAPI::Region::getRegionGroupID()]=0;
		own_region_num.insert(std::map<int, int>::value_type(currentRegion->getRegionGroupID(), 0));
		opponent_region_num.insert(std::map<int, int>::value_type(currentRegion->getRegionGroupID(), 0));
	}
	//全てのユニットのRegion情報を足す
	BOOST_FOREACH(BWAPI::Unit * currentUnit, BWAPI::Broodwar->enemy()->getUnits()){
		opponent_region_num[currentUnit->getRegion()->getRegionGroupID()] ++;
	}
	BOOST_FOREACH(BWAPI::Unit * currentUnit, BWAPI::Broodwar->self()->getUnits()){
		own_region_num[currentUnit->getRegion()->getRegionGroupID()] ++;
	}
	//各Regionごとに味方ユニット数が1つ、2つ、3つ、4つ以上で場合わけ
	std::vector<float>	temp(num_states, 0.0);
	int count_temp = 0;
	int value;
	//mapのforeachをまわしてregionごとのUnit数をvectorに変換
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
	
	//ローカル変数の状態をメンバ変数とスワップ
	states.swap(temp);

}

int Neural::setNumState(){
	//Regionの数を返す
	int region_count = 0;
	BOOST_FOREACH(BWAPI::Region * currentRegion, BWAPI::Broodwar->getAllRegions()){
		region_count++;
	}
	return region_count;
}

const std::vector<float> & Neural::getActions(){
	return inputs.back();
}