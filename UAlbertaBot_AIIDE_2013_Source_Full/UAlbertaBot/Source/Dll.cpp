#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#include <windows.h>
#include <stdio.h>
#include <tchar.h>

#include <BWAPI.h>
#include <BWTA.h>
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

#include "UAlbertaBotModule.h"
namespace BWAPI { Game* Broodwar; }
BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved
					 )
{
    
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
			BWAPI::BWAPI_init();
			break;
		case DLL_PROCESS_DETACH:
			break;
	}

	return TRUE;
}

 extern "C" __declspec(dllexport) BWAPI::AIModule* newAIModule(BWAPI::Game* game)
{
	BWAPI::Broodwar=game;
	return new UAlbertaBotModule();
}