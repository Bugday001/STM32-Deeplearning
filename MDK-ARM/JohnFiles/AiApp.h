#ifndef _AIAPP_H
#define _AIAPP_H
#include "ai_datatypes_defines.h"
#include "main.h"
void Prediction(void* network, void *in_data, void *out_data);
void John_Ai_init(void* network,ai_network_params* ai_params);
extern float imgdata[];
#endif
