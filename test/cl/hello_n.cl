__constant char message[] = "Hello, World!\0";

__kernel void hello(__global char* string)
{
	__private int id = get_local_id(0);
	//printf("\n I have an ID, I do %d",id);
	string[id] = message[id];
}

