#include "Common.h"
#include "Neural.h"


// constructor
Neural::Neural()
:selfRace(BWAPI::Broodwar->self()->getRace())
,enemyRace(BWAPI::Broodwar->enemy()->getRace())
{
	count = 0;
	createNetwork();
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
	// if the game ended before the tournament time limit
	// •ñV‚É‚æ‚Á‚Äoutputs‚ğXV‚·‚é
	if (BWAPI::Broodwar->getFrameCount() < Options::Tournament::GAME_END_FRAME)
	{
		if (isWinner)
		{
			//•ñV‚Ìİ’è
			for (float i = 0; i < count; ++i){
				outputs[i][0] = outputs[i][0] + (i/(float)count);
			}
		}
		else
		{
			for (float i = 0; i < count; ++i){
				outputs[i][0] = outputs[i][0] - (i/(float)count);
			}
		}
	}
	// otherwise game timed out so use in-game score
	// ”»’èŸ‚¿
	else
	{
		if (getScore(BWAPI::Broodwar->self()) > getScore(BWAPI::Broodwar->enemy()))
		{
			//•ñV‚Ìİ’è
			for (float i = 0; i < count; ++i){
				outputs[i][0] = outputs[i][0] + (i / (float)count);
			}
		}
		else
		{
			for (float i = 0; i < count; ++i){
				outputs[i][0] = outputs[i][0] - (i / (float)count);
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
	selectBestAction();
	srand((unsigned int)time(NULL));
	//ƒCƒvƒVƒƒ“=0.2‚Åƒ‰ƒ“ƒ_ƒ€‰»
	if (rand()%10>8){
		for (int j; j < num_actions; ++j){
			int flag = ((rand()%(int)pow(2.0, num_actions)) >> j) % 2;
			flag == 1 ? inputs[count][j] = 1.0 : inputs[count][j] = 0.0;
		}
	}
	else{
		inputs[count] = bestinput;
	}


}

void Neural::selectBestAction(){

	fann_type *best_out;
	std::vector<float> 	actions(num_actions,0.0);
	std::vector<float>	states = getState();
	std::vector<float>	input;
	//2‚Ìnum_actionsæ‚É‚Â‚¢‚Ä‘“–‚è
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


std::vector<float> &  Neural::getActions(){
	return inputs[count];
}

void Neural::update(std::set<BWAPI::Unit *> unitsToAssign){
	if (neuralUpdateFrame()){
		setState(unitsToAssign);
		setActions();

	}

}

bool Neural::neuralUpdateFrame()
{
	return BWAPI::Broodwar->getFrameCount() % 48 == 0;
}

void Neural::setState(std::set<BWAPI::Unit *> & unitsToAssign){
	std::set<BWAPI::Region*> AllRegion = BWAPI::Broodwar->getAllRegions();
	BOOST_FOREACH(BWAPI::Region * myRegion,AllRegion){
		states[AllRegion->BWAPI::Region::getRegionGroupID];
	}

}

std::vector<float>	& getState(){
	std::vector<float>	states(num_states, 0.0);
	return states;
}